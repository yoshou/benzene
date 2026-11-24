#include "dsp/models/superpoint/superpoint.h"

#include <atomic>
#include <cmath>
#include <cstring>

#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/vector.h"
#include "dsp/kernels/convolution_int8/convolution_int8.h"
#include "dsp/models/shared/cell_heatmap.h"
#include "dsp/models/shared/feature_slots.h"
#include "dsp/models/shared/layer_weights.h"
#include "dsp/runtime/cache.h"
#include "dsp/runtime/heap.h"
#include "dsp/runtime/stage_timer.h"
#include "dsp/runtime/status.h"
#include "dsp/runtime/synchronization.h"
#include "dsp/runtime/thread_pool.h"
#include "dsp/vector_math/functions.h"
#include "benzene/protocol.h"

namespace benzene::models::superpoint {
namespace {

namespace convolution = protocol::superpoint::convolution;
namespace stat = protocol::superpoint::stat;
using hvx::Mask;
using hvx::Vector;
using hvx::VectorPair;
namespace convolution_int8 = kernels::convolution_int8;

constexpr int kCompactWidth = 64;
constexpr int kBorder = 4;

constexpr int kLocateItemKeypoints = 32;

struct KeypointCells {
    int cells[4];
    float weights[4];
};

struct Workspace {
    int height;
    int width;
    runtime::HeapArray<uint8_t> image;
    runtime::HeapArray<uint8_t> maps[8];
    runtime::HeapArray<uint8_t> detector_map;
    runtime::HeapArray<uint8_t> descriptor_map;
    runtime::HeapArray<float> logits;
    runtime::HeapArray<uint8_t> compact_rows;
    runtime::HeapArray<int> cell_rows;  // cell -> compact row (-1: not sampled), all -1 between calls
    runtime::HeapArray<int> row_cells;
    runtime::GrowingArray<KeypointCells> keypoint_cells;
    runtime::HeapArray<float> candidate_scores[runtime::kHvxThreads];
    runtime::HeapArray<int> candidate_indices[runtime::kHvxThreads];
    int candidate_counts[runtime::kHvxThreads];
    int candidate_capacity;
};

runtime::Status allocate_workspace(Workspace & workspace, int height, int width) {
    if (workspace.height == height && workspace.width == width) return {};

    workspace = Workspace{};
    const size_t pixels = static_cast<size_t>(height) * width;
    const size_t cells = pixels / 64;
    const size_t slack = 4 * 256;
    const size_t map_bytes[8] = {pixels * 64,       pixels / 4 * 64,   pixels / 4 * 64,   pixels / 16 * 64,
                                 pixels / 16 * 128, pixels / 64 * 128, pixels / 64 * 128, pixels / 64 * 128};

    workspace.image = runtime::allocate_array<uint8_t>(pixels + 256);
    bool allocated = static_cast<bool>(workspace.image);
    for (int i = 0; i < 8; i++) {
        workspace.maps[i] = runtime::allocate_array<uint8_t>(map_bytes[i] + slack);
        allocated = allocated && workspace.maps[i];
    }
    workspace.detector_map = runtime::allocate_array<uint8_t>(cells * 256 + slack);
    workspace.descriptor_map = runtime::allocate_array<uint8_t>(cells * 256 + slack);
    workspace.logits = runtime::allocate_array<float>(cells * 96 + 128);
    workspace.compact_rows = runtime::allocate_array<uint8_t>((cells + kCompactWidth) * 256);
    workspace.cell_rows = runtime::allocate_array<int>(cells);
    workspace.row_cells = runtime::allocate_array<int>(cells);
    allocated = allocated && workspace.detector_map && workspace.descriptor_map && workspace.logits &&
                workspace.compact_rows && workspace.cell_rows && workspace.row_cells;

    workspace.candidate_capacity = static_cast<int>(pixels / 4);
    for (int thread = 0; thread < runtime::kHvxThreads; thread++) {
        workspace.candidate_scores[thread] = runtime::allocate_array<float>(workspace.candidate_capacity);
        workspace.candidate_indices[thread] = runtime::allocate_array<int>(workspace.candidate_capacity);
        allocated = allocated && workspace.candidate_scores[thread] && workspace.candidate_indices[thread];
    }

    if (!allocated) {
        workspace = Workspace{};
        return std::unexpected(runtime::Error::kNoMemory);
    }

    memset(workspace.cell_rows.get(), 0xff, cells * 4);
    workspace.height = height;
    workspace.width = width;
    return {};
}

// simple_nms, pool = the max over the 9 x 9 window (zeros beyond the map: every map is >= 0):
//   kInitialMaxima  maxima = (scores == pool(scores))
//   kSuppression    pooled = pool(maxima); suppressed = pooled > 0 ? 0 : scores
//   kNewMaxima      maxima |= (suppressed == pool(suppressed)) & !(pooled > 0)
enum NmsPass : int { kInitialMaxima, kSuppression, kNewMaxima };

struct NmsJob {
    NmsPass pass;
    int height;
    int stride;
    float * scores;
    float * maxima;
    float * suppressed;
    float * pooled;
    float * rings[runtime::kHvxThreads];  // per thread: kRingRows rows of ring_stride(stride), a zero row, a spare row
};

constexpr int kRingRows = 10;

// The ring's row stride in floats: rows 512 B apart would share VTCM banks.
constexpr int ring_stride(int stride) {
    return stride + 32;
}

// out[x] = max(a[x - 4 .. x + 4]) of two rows (zeros beyond) = max(b3[x - 4], a[x + 4]), where b1*, b2*, b3* are the
// forward maxima of 2, 4 and 8 lanes.
[[gnu::always_inline]] inline void horizontal_pool_step(Vector<float> * __restrict out0,
                                                        Vector<float> * __restrict out1, Vector<float> a4[2],
                                                        Vector<float> a0[2], Vector<float> a1[2], Vector<float> a2[2],
                                                        Vector<float> a3[2], Vector<float> b12[2], Vector<float> b21[2],
                                                        Vector<float> b30[2], Vector<float> b3p[2]) {
    Vector<float> b13[2];
    Vector<float> b22[2];
    Vector<float> b31[2];
    *out0 = max(slide_up(b30[0], b3p[0], 16), slide_down(a1[0], a0[0], 16));
    *out1 = max(slide_up(b30[1], b3p[1], 16), slide_down(a1[1], a0[1], 16));

    for (int r = 0; r < 2; r++) b13[r] = max(a3[r], slide_down(a4[r], a3[r], 4));
    for (int r = 0; r < 2; r++) b22[r] = max(b12[r], slide_down(b13[r], b12[r], 8));
    for (int r = 0; r < 2; r++) b31[r] = max(b21[r], slide_down(b22[r], b21[r], 16));

    for (int r = 0; r < 2; r++) {
        b3p[r] = b30[r];
        b30[r] = b31[r];
        b21[r] = b22[r];
        b12[r] = b13[r];
        a0[r] = a1[r];
        a1[r] = a2[r];
        a2[r] = a3[r];
        a3[r] = a4[r];
    }
}

[[gnu::noinline]] void pool_rows_horizontally(const Vector<float> * const * rows, Vector<float> * const * out,
                                              int vectors) {
    const Vector<float> zero = Vector<float>::zero();
    const Vector<float> * __restrict row0 = rows[0];
    const Vector<float> * __restrict row1 = rows[1];
    Vector<float> * __restrict out0 = out[0];
    Vector<float> * __restrict out1 = out[1];
    Vector<float> a0[2];
    Vector<float> a1[2];
    Vector<float> a2[2];
    Vector<float> a3[2];
    Vector<float> a4[2];
    Vector<float> b12[2];
    Vector<float> b21[2];
    Vector<float> b30[2];
    Vector<float> b3p[2];

    for (int r = 0; r < 2; r++) {
        const Vector<float> * __restrict row = r != 0 ? row1 : row0;
        a0[r] = row[0];
        a1[r] = vectors > 1 ? row[1] : Vector<float>::zero();
        a2[r] = vectors > 2 ? row[2] : Vector<float>::zero();
        a3[r] = vectors > 3 ? row[3] : Vector<float>::zero();

        const Vector<float> b1m = max(zero, slide_down(a0[r], zero, 4));
        const Vector<float> b10 = max(a0[r], slide_down(a1[r], a0[r], 4));
        const Vector<float> b11 = max(a1[r], slide_down(a2[r], a1[r], 4));
        b12[r] = max(a2[r], slide_down(a3[r], a2[r], 4));
        const Vector<float> b2m = max(b1m, slide_down(b10, b1m, 8));
        const Vector<float> b20 = max(b10, slide_down(b11, b10, 8));
        b21[r] = max(b11, slide_down(b12[r], b11, 8));
        b3p[r] = max(b2m, slide_down(b20, b2m, 16));
        b30[r] = max(b20, slide_down(b21[r], b20, 16));
    }

    int v = 0;
    for (; v + 4 < vectors; v++) {
        a4[0] = row0[v + 4];
        a4[1] = row1[v + 4];
        horizontal_pool_step(out0 + v, out1 + v, a4, a0, a1, a2, a3, b12, b21, b30, b3p);
    }
    for (; v < vectors; v++) {
        a4[0] = zero;
        a4[1] = zero;
        horizontal_pool_step(out0 + v, out1 + v, a4, a0, a1, a2, a3, b12, b21, b30, b3p);
    }
}

[[gnu::always_inline]] inline void window_maxima(const Vector<float> * const * window, const Vector<float> * last_row,
                                                 int v, int two_rows, Vector<float> * first_max,
                                                 Vector<float> * second_max) {
    const Vector<float> shared = max(max(max(window[1][v], window[2][v]), max(window[3][v], window[4][v])),
                                     max(max(window[5][v], window[6][v]), max(window[7][v], window[8][v])));
    *first_max = max(shared, window[0][v]);
    *second_max = two_rows ? max(shared, last_row[v]) : *first_max;
}

void nms_band(void * job_pointer, int thread) {
    const auto * job = static_cast<const NmsJob *>(job_pointer);
    const int height = job->height;
    const int first_y = height * thread / runtime::kHvxThreads;
    const int end_y = height * (thread + 1) / runtime::kHvxThreads;
    const int vectors = job->stride / 32;
    const int ring_row_stride = ring_stride(job->stride);
    const Vector<float> zero = Vector<float>::zero();
    const Vector<float> one = Vector<float>::splat(1.0f);
    const float * source = job->pass == kInitialMaxima ? job->scores
                           : job->pass == kSuppression ? job->maxima
                                                       : job->suppressed;
    float * ring = job->rings[thread];
    const auto * zero_row = hvx::vectors(ring + static_cast<size_t>(kRingRows) * ring_row_stride);
    auto * spare_row = hvx::vectors(ring + static_cast<size_t>(kRingRows + 1) * ring_row_stride);
    int next = first_y - 4 > 0 ? first_y - 4 : 0;

    for (int y = first_y; y < end_y; y += 2) {
        const int needed = y + 5 < height - 1 ? y + 5 : height - 1;
        while (next <= needed) {
            const int two = next + 1 <= needed;
            const Vector<float> * in[2] = {
                hvx::vectors(source + static_cast<size_t>(next) * job->stride),
                two != 0 ? hvx::vectors(source + static_cast<size_t>(next + 1) * job->stride) : zero_row};
            Vector<float> * const out[2] = {
                hvx::vectors(ring + static_cast<size_t>(next % kRingRows) * ring_row_stride),
                two != 0 ? hvx::vectors(ring + static_cast<size_t>((next + 1) % kRingRows) * ring_row_stride)
                         : spare_row};
            pool_rows_horizontally(in, out, vectors);
            next += two != 0 ? 2 : 1;
        }

        const Vector<float> * window[10];
        for (int r = 0; r < 10; r++) {
            window[r] = y - 4 + r >= 0 && y - 4 + r < height
                            ? hvx::vectors(ring + static_cast<size_t>((y - 4 + r) % kRingRows) * ring_row_stride)
                            : zero_row;
        }

        const int two = y + 1 < end_y;
        const size_t offset0 = static_cast<size_t>(y) * job->stride;
        const size_t offset1 = two != 0 ? offset0 + job->stride : offset0;
        const auto * scores0 = hvx::vectors(job->scores + offset0);
        const auto * scores1 = hvx::vectors(job->scores + offset1);
        auto * maxima0 = hvx::vectors(job->maxima + offset0);
        auto * maxima1 = hvx::vectors(job->maxima + offset1);
        auto * suppressed0 = hvx::vectors(job->suppressed + offset0);
        auto * suppressed1 = hvx::vectors(job->suppressed + offset1);
        auto * pooled0 = hvx::vectors(job->pooled + offset0);
        auto * pooled1 = hvx::vectors(job->pooled + offset1);
        const Vector<float> * last_row = two != 0 ? window[9] : window[0];

        if (job->pass == kInitialMaxima) {
            for (int v = 0; v < vectors; v++) {
                Vector<float> max0;
                Vector<float> max1;
                window_maxima(window, last_row, v, two, &max0, &max1);
                maxima0[v] = select(reinterpret<int32_t>(scores0[v]) == reinterpret<int32_t>(max0), one, zero);
                maxima1[v] = select(reinterpret<int32_t>(scores1[v]) == reinterpret<int32_t>(max1), one, zero);
            }
        } else if (job->pass == kSuppression) {
            for (int v = 0; v < vectors; v++) {
                Vector<float> max0;
                Vector<float> max1;
                window_maxima(window, last_row, v, two, &max0, &max1);
                pooled0[v] = max0;
                suppressed0[v] = select(vector_math::positive(max0), zero, scores0[v]);
                pooled1[v] = max1;
                suppressed1[v] = select(vector_math::positive(max1), zero, scores1[v]);
            }
        } else {
            for (int v = 0; v < vectors; v++) {
                Vector<float> max0;
                Vector<float> max1;
                window_maxima(window, last_row, v, two, &max0, &max1);
                const Mask new0 = and_not(reinterpret<int32_t>(suppressed0[v]) == reinterpret<int32_t>(max0),
                                          vector_math::positive(pooled0[v]));
                const Mask new1 = and_not(reinterpret<int32_t>(suppressed1[v]) == reinterpret<int32_t>(max1),
                                          vector_math::positive(pooled1[v]));
                maxima0[v] = select(new0, one, maxima0[v]);
                maxima1[v] = select(new1, one, maxima1[v]);
            }
        }
    }
}

struct CandidateJob {
    const float * scores;
    const float * maxima;
    int height;
    int width;
    int stride;
    int border;
    float threshold;
    float * rows[runtime::kHvxThreads];  // per thread: a row copy in cached memory (scalar loads from VTCM are slow)
    Workspace * workspace;
};

void collect_candidates(void * job_pointer, int thread) {
    const auto * job = static_cast<const CandidateJob *>(job_pointer);
    Workspace & workspace = *job->workspace;
    const int first_y = job->height * thread / runtime::kHvxThreads;
    const int end_y = job->height * (thread + 1) / runtime::kHvxThreads;
    const int border = job->border;
    const int width = job->width;
    const int vectors = job->stride / 32;
    const int capacity = workspace.candidate_capacity;
    const float threshold = job->threshold;
    const Vector<float> threshold_vector = Vector<float>::splat(threshold);
    float * row = job->rows[thread];
    float * candidate_scores = workspace.candidate_scores[thread].get();
    alignas(128) int32_t marked_lanes[32];
    int * candidate_indices = workspace.candidate_indices[thread].get();
    int count = 0;

    for (int y = first_y > border ? first_y : border; y < end_y && y < job->height - border; y++) {
        const auto * scores = hvx::vectors(job->scores + static_cast<size_t>(y) * job->stride);
        const auto * maxima = hvx::vectors(job->maxima + static_cast<size_t>(y) * job->stride);
        Vector<int32_t> marks = Vector<int32_t>::zero();
        for (int v = 0; v < vectors; v++) {
            const Vector<float> x = select(vector_math::positive(maxima[v]), scores[v], Vector<float>::zero());
            hvx::vectors(row)[v] = x;
            marks = marks | pattern_where<int32_t>(x > threshold_vector, 0x01010101);
        }
        store(marked_lanes, marks);

        int lanes[32];
        int lane_count = 0;
        for (int lane = 0; lane < 32; lane++) {
            if (marked_lanes[lane]) lanes[lane_count++] = lane;
        }

        for (int v = 0; v < vectors && lane_count; v++) {
            for (int k = 0; k < lane_count; k++) {
                const int x = 32 * v + lanes[k];
                const float value = row[x];
                if (x >= border && x < width - border && value > threshold && count < capacity) {
                    candidate_scores[count] = value;
                    candidate_indices[count++] = y * width + x;
                }
            }
        }
    }

    workspace.candidate_counts[thread] = count;
}

[[gnu::noinline]] float kth_largest(float * values, int count, int k) {
    int low = 0;
    int high = count - 1;
    const int target = k - 1;

    while (low < high) {
        const float pivot = values[(low + high) / 2];
        int i = low;
        int j = high;
        while (i <= j) {
            while (values[i] > pivot) i++;
            while (values[j] < pivot) j--;
            if (i <= j) {
                const float swapped = values[i];
                values[i++] = values[j];
                values[j--] = swapped;
            }
        }

        if (target <= j) {
            high = j;
        } else if (target >= i) {
            low = i;
        } else {
            return values[target];
        }
    }
    return values[target];
}

struct DescriptorJob {
    float * cell_descriptors;
    const int * cell_rows;
    const float * keypoints;
    KeypointCells * keypoint_cells;
    _Float16 * descriptors;
    int cell_rows_count;
    int cell_columns;
    int keypoint_count;
    int compact_cells;
};

// 1 / max(sqrt(lane 0 of sums[i]), 1e-12) in every lane (F.normalize's eps).
inline void reciprocal_square_roots(const Vector<float> * sums, Vector<float> * reciprocals, int count) {
    alignas(128) float lanes[4][32];
    for (int i = 0; i < count; i++) hvx::vector_at<float>(lanes[i]) = sums[i];
    for (int i = 0; i < count; i++) {
        reciprocals[i] = Vector<float>::splat(1.0f / sqrtf(lanes[i][0] > 1e-24f ? lanes[i][0] : 1e-24f));
    }
}

// As sample_descriptors (s = 8, align_corners): grid = ((k - s / 2 + 0.5) / (dim * s - s / 2 - 0.5)) * 2 - 1.
void locate_keypoint_cells(void * context, int item, int) {
    const auto * job = static_cast<const DescriptorJob *>(context);
    const int first = kLocateItemKeypoints * item;

    for (int i = first; i < first + kLocateItemKeypoints && i < job->keypoint_count; i++) {
        const float x = job->keypoints[2 * i];
        const float y = job->keypoints[2 * i + 1];
        const float ix = (x - 3.5f) / (job->cell_columns * 8 - 4.5f) * (job->cell_columns - 1);
        const float iy = (y - 3.5f) / (job->cell_rows_count * 8 - 4.5f) * (job->cell_rows_count - 1);

        const int x0 = static_cast<int>(floorf(ix));
        const int y0 = static_cast<int>(floorf(iy));
        const float fx = ix - x0;
        const float fy = iy - y0;
        const float weights[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};

        KeypointCells * entry = &job->keypoint_cells[i];
        for (int corner = 0; corner < 4; corner++) {
            const int cell_x = x0 + corner % 2;
            const int cell_y = y0 + corner / 2;
            entry->cells[corner] =
                cell_x >= 0 && cell_x < job->cell_columns && cell_y >= 0 && cell_y < job->cell_rows_count
                    ? cell_y * job->cell_columns + cell_x
                    : -1;
            entry->weights[corner] = weights[corner];
        }
    }
}

void compact_cell_row(void * context, int row, int) {
    const Workspace & workspace = *static_cast<const Workspace *>(context);
    const auto * source =
        hvx::vectors(workspace.descriptor_map.get() + static_cast<size_t>(workspace.row_cells[row]) * 256);
    auto * destination = hvx::vectors(workspace.compact_rows.get() + static_cast<size_t>(row) * 256);
    destination[0] = source[0];
    destination[1] = source[1];
}

void normalize_cells(void * context, int item, int) {
    const auto * job = static_cast<const DescriptorJob *>(context);
    const int count = job->compact_cells - 4 * item < 4 ? job->compact_cells - 4 * item : 4;
    Vector<float> * cells[4];
    Vector<float> sums[4];
    Vector<float> reciprocals[4];
    for (int c = 0; c < 4; c++) {
        cells[c] = hvx::vectors(job->cell_descriptors + static_cast<size_t>(4 * item + (c < count ? c : 0)) * 256);
        sums[c] = Vector<float>::zero();
    }

    for (int k = 0; k < 8; k++) {
        for (int c = 0; c < 4; c++) sums[c] = sums[c] + cells[c][k] * cells[c][k];
    }
    for (int c = 0; c < 4; c++) sums[c] = vector_math::sum_to_all_lanes(sums[c]);
    reciprocal_square_roots(sums, reciprocals, count);

    for (int c = 0; c < count; c++) {
        for (int k = 0; k < 8; k++) cells[c][k] = cells[c][k] * reciprocals[c];
    }
}

void sample_descriptors(void * context, int item, int) {
    const auto * job = static_cast<const DescriptorJob *>(context);
    const int count = job->keypoint_count - 2 * item < 2 ? job->keypoint_count - 2 * item : 2;
    Vector<float> sums_of_corners[2][8];
    Vector<float> sums[2];
    Vector<float> reciprocals[2];

    for (int p = 0; p < 2; p++) {
        for (int k = 0; k < 8; k++) sums_of_corners[p][k] = Vector<float>::zero();
    }
    for (int p = 0; p < count; p++) {
        const KeypointCells * entry = &job->keypoint_cells[2 * item + p];
        for (int corner = 0; corner < 4; corner++) {
            if (entry->cells[corner] < 0) continue;
            const auto * cell =
                hvx::vectors(job->cell_descriptors + static_cast<size_t>(job->cell_rows[entry->cells[corner]]) * 256);
            const Vector<float> weight = Vector<float>::splat(entry->weights[corner]);
            for (int k = 0; k < 8; k++) sums_of_corners[p][k] = sums_of_corners[p][k] + cell[k] * weight;
        }
    }

    sums[0] = Vector<float>::zero();
    sums[1] = Vector<float>::zero();
    for (int k = 0; k < 8; k++) {
        for (int p = 0; p < 2; p++) sums[p] = sums[p] + sums_of_corners[p][k] * sums_of_corners[p][k];
    }
    for (int p = 0; p < 2; p++) sums[p] = vector_math::sum_to_all_lanes(sums[p]);
    reciprocal_square_roots(sums, reciprocals, count);

    for (int p = 0; p < count; p++) {
        auto * out = hvx::vectors(job->descriptors + static_cast<size_t>(2 * item + p) * 256);
        for (int k = 0; k < 4; k++) {
            const VectorPair<float> dealt = deinterleave(sums_of_corners[p][2 * k + 1] * reciprocals[p],
                                                         sums_of_corners[p][2 * k] * reciprocals[p]);
            out[k] = to_float16_interleaved(dealt.low(), dealt.high());
        }
    }
}

// One call at a time: the calls share the session's state.
struct CallGuard {
    std::atomic<int> in_use;
};

}  // namespace

runtime::Status run(runtime::Session & session, const Arguments & arguments) {
    const int height = arguments.height;
    const int width = arguments.width;
    const int max_keypoints = arguments.max_keypoints;
    int32_t * const statistics = arguments.statistics;
    Features * features = feature_slot(session, arguments.slot);
    const int model_slot = arguments.references[0];
    if (!session.model_mapped(model_slot) || height % 8 != 0 || width % 8 != 0 || height < 16 || width < 16 ||
        max_keypoints < 1 || features == nullptr) {
        return std::unexpected(runtime::Error::kBadParameter);
    }

    runtime::Status status = reserve_features(features, max_keypoints, protocol::superpoint::kDescriptorDimensions);
    if (!status) return status;
    features->count = 0;
    features->height = height;
    features->width = width;
    const LayerWeights model{session.model_base(model_slot), arguments.references + 1};
    runtime::StageTimer timer(statistics);

    CallGuard * guard = session.state<CallGuard>();
    if (guard == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const runtime::ExclusiveUse use(guard->in_use);
    if (!use.taken()) return std::unexpected(runtime::Error::kBadParameter);
    Workspace * kept = session.state<Workspace>();
    if (kept == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    Workspace & workspace = *kept;
    status = allocate_workspace(workspace, height, width);
    if (!status) return status;
    runtime::invalidate_host_input(arguments.image, static_cast<size_t>(height) * width);
    status = runtime::copy_memory_parallel(session, workspace.image.get(), arguments.image,
                                           static_cast<size_t>(height) * width);
    if (!status) return status;

    uint8_t * maps[8];
    for (int i = 0; i < 8; i++) maps[i] = workspace.maps[i].get();
    const int cell_rows = height / 8;
    const int cell_columns = width / 8;

    struct Layer {
        int convolution;
        int height;
        int width;
        int channels;
        int taps;
        int pool;
        int N;
        const uint8_t * input;
        uint8_t * output;
    };
    const Layer layers[8] = {
        {convolution::kConv1a, height, width, 1, 9, 0, 64, workspace.image.get(), maps[0]},
        {convolution::kConv1b, height, width, 64, 9, 1, 64, maps[0], maps[1]},
        {convolution::kConv2a, height / 2, width / 2, 64, 9, 0, 64, maps[1], maps[2]},
        {convolution::kConv2b, height / 2, width / 2, 64, 9, 1, 64, maps[2], maps[3]},
        {convolution::kConv3a, height / 4, width / 4, 64, 9, 0, 128, maps[3], maps[4]},
        {convolution::kConv3b, height / 4, width / 4, 128, 9, 1, 128, maps[4], maps[5]},
        {convolution::kConv4a, cell_rows, cell_columns, 128, 9, 0, 128, maps[5], maps[6]},
        {convolution::kConv4b, cell_rows, cell_columns, 128, 9, 0, 128, maps[6], maps[7]},
    };
    for (int l = 0; l < 8; l++) {
        convolution_int8::Arguments call;
        call.input = layers[l].input;
        call.height = layers[l].height;
        call.width = layers[l].width;
        call.channels = layers[l].channels;
        call.taps = layers[l].taps;
        call.pool = layers[l].pool;
        call.weights = model.part<uint8_t>(layers[l].convolution);
        call.N = layers[l].N;
        call.uint8_outputs[0] = layers[l].output;
        call.uint8_output_channels = layers[l].N;
        status = kernels::convolution_int8::run(session, call);
        if (!status) return status;

        timer.lap(stat::kFirstConvolutionMicroseconds + l);
    }

    {
        convolution_int8::Arguments call;
        call.input = maps[7];
        call.height = cell_rows;
        call.width = cell_columns;
        call.channels = 128;
        call.taps = 9;
        call.weights = model.part<uint8_t>(convolution::kDetectorAndDescriptorA);
        call.N = 512;
        call.uint8_outputs[0] = workspace.detector_map.get();
        call.uint8_outputs[1] = workspace.descriptor_map.get();
        call.uint8_output_channels = 256;
        status = kernels::convolution_int8::run(session, call);
        if (!status) return status;
    }
    {
        convolution_int8::Arguments call;
        call.input = workspace.detector_map.get();
        call.height = cell_rows;
        call.width = cell_columns;
        call.channels = 256;
        call.taps = 1;
        call.weights = model.part<uint8_t>(convolution::kDetectorB);
        call.N = 96;
        call.output_form = convolution_int8::OutputForm::kRowsF32;
        call.rows_f32 = workspace.logits.get();
        status = kernels::convolution_int8::run(session, call);
        if (!status) return status;
    }
    timer.lap(stat::kHeadsMicroseconds);

    const int stride = (width + 31) / 32 * 32;
    const size_t map_bytes = static_cast<size_t>(height) * stride * 4;
    const size_t ring_bytes = static_cast<size_t>(kRingRows + 2) * ring_stride(stride) * 4;
    if (4 * map_bytes + 4 * ring_bytes > session.vtcm_bytes()) return std::unexpected(runtime::Error::kNoMemory);
    uint8_t * vtcm = session.vtcm();
    auto * heatmap = reinterpret_cast<float *>(vtcm);
    auto * maxima = reinterpret_cast<float *>(vtcm + map_bytes);
    auto * suppressed = reinterpret_cast<float *>(vtcm + 2 * map_bytes);
    auto * pooled = reinterpret_cast<float *>(vtcm + 3 * map_bytes);

    if (stride > width) {
        for (int y = 0; y < height; y++) {
            hvx::vectors(heatmap + static_cast<size_t>(y) * stride)[stride / 32 - 1] = Vector<float>::zero();
        }
    }
    CellHeatmapJob score_job{workspace.logits.get(), heatmap, cell_columns, stride};
    status = runtime::parallel_for(session, runtime::kHvxThreads, cell_rows, softmax_cell_row, &score_job);
    if (!status) return status;
    timer.lap(stat::kSoftmaxMicroseconds);

    NmsJob nms_job{kInitialMaxima, height, stride, heatmap, maxima, suppressed, pooled, {}};
    for (int thread = 0; thread < runtime::kHvxThreads; thread++) {
        nms_job.rings[thread] =
            reinterpret_cast<float *>(vtcm + 4 * map_bytes + static_cast<size_t>(thread) * ring_bytes);
        auto * zero_row = hvx::vectors(nms_job.rings[thread] + static_cast<size_t>(kRingRows) * ring_stride(stride));
        for (int v = 0; v < ring_stride(stride) / 32; v++) zero_row[v] = Vector<float>::zero();
    }
    status = runtime::run_parallel(session, runtime::kHvxThreads, nms_band, &nms_job);
    if (!status) return status;
    for (int round = 0; round < 2; round++) {
        nms_job.pass = kSuppression;
        status = runtime::run_parallel(session, runtime::kHvxThreads, nms_band, &nms_job);
        if (!status) return status;
        nms_job.pass = kNewMaxima;
        status = runtime::run_parallel(session, runtime::kHvxThreads, nms_band, &nms_job);
        if (!status) return status;
    }
    timer.lap(stat::kNmsMicroseconds);

    CandidateJob candidate_job{heatmap, maxima, height, width, stride, kBorder, arguments.threshold, {}, &workspace};
    const runtime::HeapArray<float> rows = runtime::allocate_array<float>(static_cast<size_t>(4) * stride);
    if (!rows) return std::unexpected(runtime::Error::kNoMemory);
    for (int thread = 0; thread < runtime::kHvxThreads; thread++)
        candidate_job.rows[thread] = rows.get() + static_cast<size_t>(thread) * stride;
    status = runtime::run_parallel(session, runtime::kHvxThreads, collect_candidates, &candidate_job);
    if (!status) return status;
    int candidates = 0;
    for (int thread = 0; thread < runtime::kHvxThreads; thread++) candidates += workspace.candidate_counts[thread];
    statistics[stat::kCandidates] = candidates;

    float kth_score = 0.0f;
    int ties_to_take = 0;
    if (candidates > max_keypoints) {
        const runtime::HeapArray<float> all_scores = runtime::allocate_array<float>(candidates);
        if (!all_scores) return std::unexpected(runtime::Error::kNoMemory);
        int offset = 0;
        for (int thread = 0; thread < runtime::kHvxThreads; thread++) {
            memcpy(all_scores.get() + offset, workspace.candidate_scores[thread].get(),
                   sizeof(float) * workspace.candidate_counts[thread]);
            offset += workspace.candidate_counts[thread];
        }
        kth_score = kth_largest(all_scores.get(), candidates, max_keypoints);

        int greater = 0;
        for (int thread = 0; thread < runtime::kHvxThreads; thread++) {
            for (int k = 0; k < workspace.candidate_counts[thread]; k++) {
                greater += workspace.candidate_scores[thread][k] > kth_score;
            }
        }
        ties_to_take = max_keypoints - greater;
    }

    int count = 0;
    for (int thread = 0; thread < runtime::kHvxThreads; thread++) {
        for (int c = 0; c < workspace.candidate_counts[thread]; c++) {
            const float score = workspace.candidate_scores[thread][c];
            if (candidates > max_keypoints && !(score > kth_score || (score == kth_score && ties_to_take-- > 0))) {
                continue;
            }

            const int index = workspace.candidate_indices[thread][c];
            features->keypoints[2 * count] = static_cast<float>(index % width);
            features->keypoints[2 * count + 1] = static_cast<float>(index / width);
            arguments.scores[count++] = score;
        }
    }
    memcpy(arguments.keypoints, features->keypoints.get(), static_cast<size_t>(count) * 8);
    statistics[stat::kKeypoints] = count;
    timer.lap(stat::kKeypointSelectionMicroseconds);

    KeypointCells * const keypoint_cells = workspace.keypoint_cells.reserve(count);
    if (keypoint_cells == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    DescriptorJob descriptor_job{nullptr,
                                 workspace.cell_rows.get(),
                                 features->keypoints.get(),
                                 keypoint_cells,
                                 features->descriptors.get(),
                                 cell_rows,
                                 cell_columns,
                                 count,
                                 0};
    status =
        runtime::parallel_for(session, runtime::kHvxThreads, (count + kLocateItemKeypoints - 1) / kLocateItemKeypoints,
                              locate_keypoint_cells, &descriptor_job);
    if (!status) return status;

    int compact_cells = 0;
    for (int i = 0; i < count; i++) {
        for (int corner = 0; corner < 4; corner++) {
            const int cell = keypoint_cells[i].cells[corner];
            if (cell >= 0 && workspace.cell_rows[cell] < 0) {
                workspace.cell_rows[cell] = compact_cells;
                workspace.row_cells[compact_cells++] = cell;
            }
        }
    }
    const int compact_rows = (compact_cells + kCompactWidth - 1) / kCompactWidth;
    const size_t descriptor_bytes =
        (static_cast<size_t>(compact_rows) * kCompactWidth * 1024 + 4095) & ~static_cast<size_t>(4095);
    if (descriptor_bytes > session.vtcm_bytes()) return std::unexpected(runtime::Error::kNoMemory);
    descriptor_job.cell_descriptors = reinterpret_cast<float *>(vtcm + session.vtcm_bytes() - descriptor_bytes);
    descriptor_job.compact_cells = compact_cells;

    if (compact_cells != 0) {
        status = runtime::parallel_for(session, runtime::kHvxThreads, compact_cells, compact_cell_row, &workspace);
        if (status) {
            memset(workspace.compact_rows.get() + static_cast<size_t>(compact_cells) * 256, 0,
                   static_cast<size_t>(compact_rows * kCompactWidth - compact_cells) * 256);
            runtime::Session below_descriptors =
                session.with_vtcm_bytes(static_cast<uint32_t>(session.vtcm_bytes() - descriptor_bytes));
            convolution_int8::Arguments call;
            call.input = workspace.compact_rows.get();
            call.height = compact_rows;
            call.width = kCompactWidth;
            call.channels = 256;
            call.taps = 1;
            call.weights = model.part<uint8_t>(convolution::kDescriptorB);
            call.N = 256;
            call.output_form = convolution_int8::OutputForm::kRowsF32;
            call.rows_f32 = descriptor_job.cell_descriptors;
            status = kernels::convolution_int8::run(below_descriptors, call);
        }
        if (status) {
            status = runtime::parallel_for(session, runtime::kHvxThreads, (compact_cells + 3) / 4, normalize_cells,
                                           &descriptor_job);
        }
    }
    if (status) {
        status =
            runtime::parallel_for(session, runtime::kHvxThreads, (count + 1) / 2, sample_descriptors, &descriptor_job);
    }
    for (int i = 0; i < compact_cells; i++) workspace.cell_rows[workspace.row_cells[i]] = -1;
    if (!status) return status;

    status = runtime::copy_memory_parallel(session, arguments.descriptors, features->descriptors.get(),
                                           static_cast<size_t>(count) * 512);
    if (!status) return status;
    features->count = count;
    timer.lap(stat::kDescriptorsMicroseconds);
    return {};
}

}  // namespace benzene::models::superpoint
