#include "dsp/models/xfeat/xfeat.h"

#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <utility>

#include <hexagon_protos.h>
#include <hexagon_types.h>

#include "dsp/hvx/dot_products.h"
#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/prefetch.h"
#include "dsp/hvx/scalar_math.h"
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

namespace benzene::models::xfeat {
namespace {

namespace tensor = protocol::xfeat::tensor;
namespace stat = protocol::xfeat::stat;
using hvx::Mask;
using hvx::Vector;
using hvx::VectorPair;
namespace convolution_int8 = kernels::convolution_int8;

constexpr int kVectorBytes = 128;
constexpr int kMapSlackPixels = 8;
constexpr int kLogitColumns = 96;
constexpr int kReliabilityColumns = 32;
constexpr int kMaximumSide = 4096;

constexpr int kHeatmapInputItemCells = 64;
constexpr int kReliabilityItemCells = 256;
constexpr int kSampleItemKeypoints = 8;

constexpr size_t padded_image_row_bytes(int width) {
    return (static_cast<size_t>(width) + kVectorBytes - 1) / kVectorBytes * kVectorBytes + kVectorBytes;
}

// InterpolateSparse2d's g = 2 (k / (size - 1)) - 1, then grid_sample's ((g + 1) map_size - 1) / 2, every step
// rounded to fp32.
inline float sample_coordinate(int k, int size, int map_size) {
    const float ratio = static_cast<float>(k) / static_cast<float>(size - 1);
    const float doubled = 2.0f * ratio;
    const float grid = doubled - 1.0f;
    const float shifted = grid + 1.0f;
    const float scaled = shifted * static_cast<float>(map_size);
    const float lowered = scaled - 1.0f;
    return lowered * 0.5f;
}

struct SampleTaps {
    int nearest;
    int first;
    float first_weight;
    float second_weight;
};

SampleTaps sample_taps(int k, int size) {
    SampleTaps taps;
    taps.nearest = static_cast<int>(nearbyintf(sample_coordinate(k, size, size)));
    const float position = sample_coordinate(k, size, size / 8);
    const float first = floorf(position);
    taps.first = static_cast<int>(first);
    taps.first_weight = (first + 1.0f) - position;
    taps.second_weight = position - first;
    return taps;
}

struct Workspace {
    int height;
    int width;
    // the padded image (Block1Arguments::image), zeroed at allocation: the copies write only the pixels
    runtime::HeapArray<uint8_t> image;
    runtime::HeapArray<uint8_t> block2_input;
    runtime::HeapArray<uint8_t> maps[16];
    runtime::HeapArray<float> descriptor_map;
    runtime::HeapArray<float> reliability_rows;
    runtime::HeapArray<float> logits;
    runtime::HeapArray<float> reliability;
    runtime::HeapArray<float> keypoint_multipliers;
    runtime::HeapArray<float> keypoint_biases;
    runtime::HeapArray<SampleTaps> column_taps;
    runtime::HeapArray<SampleTaps> row_taps;
    runtime::HeapArray<float> row_buffers;
    runtime::HeapArray<float> candidate_scores[runtime::kHvxThreads];
    runtime::HeapArray<int32_t> candidate_positions[runtime::kHvxThreads];
    runtime::HeapArray<float> candidate_scratch_scores[runtime::kHvxThreads];
    runtime::HeapArray<int32_t> candidate_scratch_positions[runtime::kHvxThreads];
    const float * sorted_scores[runtime::kHvxThreads];
    const int32_t * sorted_positions[runtime::kHvxThreads];
    int candidate_counts[runtime::kHvxThreads];
    int candidate_capacity;
};

enum MapIndex : int {
    kBlock2First = 0,
    kBlock2Second,
    kBlock3First,
    kBlock3Second,
    kX3,
    kBlock4First,
    kBlock4Second,
    kX4,
    kBlock5First,
    kBlock5Second,
    kBlock5Third,
    kX5,
    kFusionInput,  // then fusion.0 / .1 reuse kBlock3First / kBlock3Second (same size)
    kHeatmapInput,
    kKeypointInput,
    kHeadScratch,  // the heads' hidden maps reuse it and kFusionInput
    kMapCount
};

runtime::Status allocate_workspace(Workspace & workspace, int height, int width) {
    if (workspace.height == height && workspace.width == width) return {};

    workspace = Workspace{};
    const size_t image_bytes = (static_cast<size_t>(height) + 2) * padded_image_row_bytes(width);
    const size_t pixels4 = static_cast<size_t>(height / 4) * (width / 4);
    const size_t cells = static_cast<size_t>(height / 8) * (width / 8);
    const size_t pixels16 = static_cast<size_t>(height / 16) * (width / 16);
    const size_t pixels32 = static_cast<size_t>(height / 32) * (width / 32);
    const size_t map_bytes[kMapCount] = {
        (pixels4 + kMapSlackPixels) * 32,   (pixels4 + kMapSlackPixels) * 32,   (cells + kMapSlackPixels) * 64,
        (cells + kMapSlackPixels) * 64,     (cells + kMapSlackPixels) * 64,     (pixels16 + kMapSlackPixels) * 64,
        (pixels16 + kMapSlackPixels) * 64,  (pixels16 + kMapSlackPixels) * 64,  (pixels32 + kMapSlackPixels) * 128,
        (pixels32 + kMapSlackPixels) * 128, (pixels32 + kMapSlackPixels) * 128, (pixels32 + kMapSlackPixels) * 64,
        (cells + kMapSlackPixels) * 64,     (cells + kMapSlackPixels) * 64,     (cells + kMapSlackPixels) * 64,
        (cells + kMapSlackPixels) * 64};

    bool allocated = runtime::allocate(workspace.image, image_bytes);
    allocated &= runtime::allocate(workspace.block2_input, (pixels4 + kMapSlackPixels) * 32 + 256);
    for (int map = 0; map < kMapCount; map++) allocated &= runtime::allocate(workspace.maps[map], map_bytes[map] + 256);
    allocated &= runtime::allocate(workspace.descriptor_map, (cells + kMapSlackPixels) * 64);
    allocated &= runtime::allocate(workspace.reliability_rows, (cells + kMapSlackPixels) * kReliabilityColumns);
    allocated &= runtime::allocate(workspace.logits, (cells + kMapSlackPixels) * kLogitColumns);
    allocated &= runtime::allocate(workspace.reliability, cells + 32);
    allocated &= runtime::allocate(workspace.keypoint_multipliers, 64);
    allocated &= runtime::allocate(workspace.keypoint_biases, 64);
    allocated &= runtime::allocate(workspace.column_taps, width);
    allocated &= runtime::allocate(workspace.row_taps, height);
    allocated &= runtime::allocate(workspace.row_buffers, static_cast<size_t>(4) * 2 * ((width + 31) / 32 * 32));

    workspace.candidate_capacity = static_cast<int>(static_cast<size_t>(height) * width / 4 / 4);
    const auto candidates = static_cast<size_t>(workspace.candidate_capacity);
    for (int thread = 0; thread < runtime::kHvxThreads; thread++) {
        allocated &= runtime::allocate(workspace.candidate_scores[thread], candidates);
        allocated &= runtime::allocate(workspace.candidate_positions[thread], candidates);
        allocated &= runtime::allocate(workspace.candidate_scratch_scores[thread], candidates);
        allocated &= runtime::allocate(workspace.candidate_scratch_positions[thread], candidates);
    }

    if (!allocated) {
        workspace = Workspace{};
        return std::unexpected(runtime::Error::kNoMemory);
    }

    memset(workspace.image.get(), 0, image_bytes);
    for (int x = 0; x < width; x++) workspace.column_taps[x] = sample_taps(x, width);
    for (int y = 0; y < height; y++) workspace.row_taps[y] = sample_taps(y, height);
    workspace.height = height;
    workspace.width = width;
    return {};
}

struct ImageCopyJob {
    const uint8_t * source;
    uint8_t * rows;
    size_t row_bytes;
    int height;
    int width;
};

void copy_image_band(void * job_pointer, int thread) {
    const auto * job = static_cast<const ImageCopyJob *>(job_pointer);
    for (int y = job->height * thread / runtime::kHvxThreads; y < job->height * (thread + 1) / runtime::kHvxThreads;
         y++) {
        memcpy(job->rows + static_cast<size_t>(y) * job->row_bytes, job->source + static_cast<size_t>(y) * job->width,
               static_cast<size_t>(job->width));
    }
}

constexpr int kOnes = 0x01010101;

constexpr int block1_padded_image_row_bytes(int width) {
    return hvx::round_up(width, kVectorBytes) + kVectorBytes;
}

struct ImageNormalization {
    float mean;
    float inverse_std;
};

// - height, width: multiples of 32.
// - image: uint8 rows round_up(width, 128) + 128 bytes apart, zero beyond width, with a zero row above and below.
// - map0 [H][W][4], map1 [H/2][W/2][8], map2 [H/2][W/2][8]: each with a zero row above and below.
// - output: block2's signed input [H/4][W/4][32] (channels 24 .. 31 at the zero point 128).
// - tensors (protocol::xfeat::tensor::kBlock1Layer0 ..): int8 weight words padded to 128 bytes, then 32-lane vectors:
//   - layer 0: words [4][3 rows] (bytes: taps dx = -1, 0, 1, then 0); fp32 k, beta; int32 tap sums S (the weights'
//     sum over the taps inside the image) of 9 border classes 3 r + c (r, c: 0 first, 1 inner, 2 last row / column);
//   - layer 1: words [8][9 taps] (bytes: input channels 0 .. 3); fp32 m, b;
//   - layer 2: words [8][9][2] (input channels 4g .. 4g + 3); m, b;
//   - layer 3: words [24][9][2]; scale, bias, the skip branch's weight / 16, weight and bias, the inverse of block2's
//     input scales.
struct Block1Arguments {
    const uint8_t * image;
    int height;
    int width;
    ImageNormalization normalization;
    const uint8_t * tensors[4];
    uint8_t * map0;
    uint8_t * map1;
    uint8_t * map2;
    uint8_t * output;
};

struct LayerTensor {
    const int32_t * weights;
    const float * constants;
};

LayerTensor layer_tensor(const uint8_t * tensor, int weight_words) {
    return {reinterpret_cast<const int32_t *>(tensor),
            reinterpret_cast<const float *>(tensor + hvx::round_up(4 * weight_words, kVectorBytes))};
}

// The saturating packs of the result clamp it to [0, 255].
inline Vector<int32_t> requantize(Vector<int32_t> acc, Vector<float> m, Vector<float> b) {
    return truncate_to_int32(to_float(acc) * m + b);
}

// Clamped first (same bytes after the packs): per-image multipliers can be large, out-of-range conversions are slow.
inline Vector<int32_t> requantize_clamped(Vector<int32_t> acc, Vector<float> m, Vector<float> b) {
    const Vector<float> v = to_float(acc) * m + b;
    return truncate_to_int32(min(max(v, -256.0f), 512.0f));
}

inline Vector<uint8_t> previous_pixels(Vector<uint8_t> current, Vector<uint8_t> previous) {
    return slide_up(current, previous, 4);
}
inline Vector<uint8_t> next_pixels(Vector<uint8_t> current, Vector<uint8_t> next) {
    return slide_down(next, current, 4);
}

struct SumJob {
    const uint8_t * rows;
    int row_bytes;
    int height;
    uint64_t sums[runtime::kHvxThreads];
    uint64_t squares[runtime::kHvxThreads];
};

void sum_band(void * job_pointer, int thread) {
    auto * job = static_cast<SumJob *>(job_pointer);
    const int first = job->height * thread / runtime::kHvxThreads;
    const int end = job->height * (thread + 1) / runtime::kHvxThreads;
    Vector<uint32_t> sums = Vector<uint32_t>::zero();
    Vector<uint32_t> squares = Vector<uint32_t>::zero();

    for (int y = first; y < end; y++) {
        const auto * row = hvx::vectors(job->rows + static_cast<size_t>(y) * job->row_bytes);
        for (int v = 0; v < job->row_bytes / kVectorBytes; v++) {
            sums = dot_product_accumulate_scalar(sums, row[v], kOnes);
            squares = dot_product_accumulate(squares, row[v], row[v]);
        }
    }

    alignas(128) uint32_t sum_lanes[32];
    alignas(128) uint32_t square_lanes[32];
    store(sum_lanes, sums);
    store(square_lanes, squares);
    uint64_t sum = 0;
    uint64_t square = 0;
    for (int lane = 0; lane < 32; lane++) {
        sum += sum_lanes[lane];
        square += square_lanes[lane];
    }
    job->sums[thread] = sum;
    job->squares[thread] = square;
}

struct Layer0Job {
    const uint8_t * image;
    int row_bytes;
    int height;
    int width;
    uint8_t * output;
    int32_t weights[4][3];
    float multipliers[4];
    float biases[4][9];
};

// words[g] lane i: the bytes (x - 1, x, x + 1, 0) of pixel x = x0 + 32 g + i of a padded image row.
inline void window_words(const uint8_t * row, int x0, Vector<uint8_t> words[4]) {
    const Vector<uint8_t> current = hvx::load(row + x0);
    const Vector<uint8_t> previous = x0 > 0 ? hvx::load(row + x0 - kVectorBytes) : Vector<uint8_t>::zero();
    const Vector<uint8_t> next = hvx::load(row + x0 + kVectorBytes);
    const Vector<uint8_t> left = slide_up(current, previous, 1);
    const Vector<uint8_t> right = slide_down(next, current, 1);

    const VectorPair<uint8_t> left_center = interleave(current, left);
    const VectorPair<uint8_t> right_zero = interleave(Vector<uint8_t>::zero(), right);
    const VectorPair<uint8_t> low = interleave_chunks<2>(right_zero.low(), left_center.low());
    const VectorPair<uint8_t> high = interleave_chunks<2>(right_zero.high(), left_center.high());
    words[0] = low.low();
    words[1] = low.high();
    words[2] = high.low();
    words[3] = high.high();
}

void layer0_row(void * context, int y, int) {
    const auto * job = static_cast<const Layer0Job *>(context);
    const int width = job->width;
    const int row_class = y == 0 ? 0 : y == job->height - 1 ? 2 : 1;

    const Mask first_lane = Mask::first_bytes(4);
    const Mask last_lane = !Mask::first_bytes(124);
    Vector<float> multipliers[4];
    Vector<float> middle[4];
    Vector<float> first[4];
    Vector<float> last[4];
    for (int n = 0; n < 4; n++) {
        multipliers[n] = Vector<float>::splat(job->multipliers[n]);
        middle[n] = Vector<float>::splat(job->biases[n][3 * row_class + 1]);
        first[n] = select(first_lane, Vector<float>::splat(job->biases[n][3 * row_class]), middle[n]);
        last[n] = select(last_lane, Vector<float>::splat(job->biases[n][3 * row_class + 2]), middle[n]);
    }

    const uint8_t * rows[3];
    for (int r = 0; r < 3; r++) rows[r] = job->image + static_cast<ptrdiff_t>(y - 1 + r) * job->row_bytes;
    auto * output = hvx::vectors(job->output + static_cast<size_t>(y) * width * 4);
    for (int x0 = 0; x0 < width; x0 += 128) {
        Vector<uint8_t> windows[3][4];
        for (int r = 0; r < 3; r++) window_words(rows[r], x0, windows[r]);

        const int groups = width - x0 >= 128 ? 4 : (width - x0) / 32;
        for (int g = 0; g < groups; g++) {
            const int x = x0 + 32 * g;
            const Vector<float> * biases = x == 0 ? first : x + 32 == width ? last : middle;
            Vector<int32_t> q[4];
#pragma clang loop unroll(full)
            for (int n = 0; n < 4; n++) {
                Vector<int32_t> acc = Vector<int32_t>::zero();
#pragma clang loop unroll(full)
                for (int r = 0; r < 3; r++) acc = dot_product_accumulate_scalar(acc, windows[r][g], job->weights[n][r]);
                q[n] = requantize_clamped(acc, multipliers[n], biases[n]);
            }
            output[x / 32] = pack_to_uint8_interleaved(q[0], q[1], q[2], q[3]);
        }
    }
}

struct EightChannelJob {
    const uint8_t * input;
    int input_width;
    uint8_t * output;
    int output_width;
    const int32_t * weights;
    Vector<float> multipliers[8];
    Vector<float> biases[8];
};

inline void store_eight_channels(const EightChannelJob * job, const Vector<int32_t> acc[8], Vector<uint8_t> * output,
                                 int pixels) {
    Vector<int32_t> q[8];
#pragma clang loop unroll(full)
    for (int n = 0; n < 8; n++) q[n] = requantize(acc[n], job->multipliers[n], job->biases[n]);
    const Vector<uint8_t> low = pack_to_uint8_interleaved(q[0], q[1], q[2], q[3]);
    const Vector<uint8_t> high = pack_to_uint8_interleaved(q[4], q[5], q[6], q[7]);
    const VectorPair<uint8_t> words = interleave_chunks<4>(high, low);
    output[0] = words.low();
    if (pixels > 16) output[1] = words.high();
}

void layer1_row(void * context, int y, int) {
    const auto * job = static_cast<const EightChannelJob *>(context);
    const size_t input_row_words = static_cast<size_t>(job->input_width);
    const int32_t * weights = job->weights;
    const Vector<uint8_t> * rows[3];
    for (int r = 0; r < 3; r++) {
        rows[r] = hvx::vectors(job->input) +
                  static_cast<ptrdiff_t>(2 * y - 1 + r) * static_cast<ptrdiff_t>(input_row_words / 32);
    }
    auto * output = hvx::vectors(job->output + static_cast<size_t>(y) * job->output_width * 8);
    Vector<uint8_t> previous_odd[3] = {Vector<uint8_t>::zero(), Vector<uint8_t>::zero(), Vector<uint8_t>::zero()};

    for (int group = 0; group * 32 < job->output_width; group++) {
        Vector<uint8_t> taps[9];
#pragma clang loop unroll(full)
        for (int r = 0; r < 3; r++) {
            const VectorPair<uint8_t> pixels = deinterleave_chunks<4>(rows[r][2 * group + 1], rows[r][2 * group]);
            const Vector<uint8_t> even = pixels.low();
            const Vector<uint8_t> odd = pixels.high();
            taps[3 * r] = previous_pixels(odd, previous_odd[r]);
            taps[3 * r + 1] = even;
            taps[3 * r + 2] = odd;
            previous_odd[r] = odd;
        }

        Vector<int32_t> acc[8];
#pragma clang loop unroll(full)
        for (int n = 0; n < 8; n++) {
            acc[n] = Vector<int32_t>::zero();
#pragma clang loop unroll(full)
            for (int t = 0; t < 9; t++) acc[n] = dot_product_accumulate_scalar(acc[n], taps[t], weights[9 * n + t]);
        }
        store_eight_channels(job, acc, output + 2 * group, job->output_width - 32 * group);
    }
}

void layer2_row(void * context, int y, int) {
    const auto * job = static_cast<const EightChannelJob *>(context);
    const int32_t * weights = job->weights;
    const int groups = (job->output_width + 31) / 32;
    const int last_pixels = job->output_width - 32 * (groups - 1);
    const Mask last_lanes{Q6_Q_vsetq2_R(4 * last_pixels)};
    const Vector<uint8_t> * rows[3];
    for (int r = 0; r < 3; r++) {
        rows[r] = hvx::vectors(job->input) +
                  static_cast<ptrdiff_t>(y - 1 + r) * static_cast<ptrdiff_t>(job->input_width / 16);
    }
    auto * output = hvx::vectors(job->output + static_cast<size_t>(y) * job->output_width * 8);

    Vector<uint8_t> previous[3][2];
    Vector<uint8_t> current[3][2];
    Vector<uint8_t> next[3][2];
#pragma clang loop unroll(full)
    for (int r = 0; r < 3; r++) {
        const VectorPair<uint8_t> words = deinterleave_chunks<4>(rows[r][1], rows[r][0]);
        previous[r][0] = Vector<uint8_t>::zero();
        previous[r][1] = Vector<uint8_t>::zero();
        current[r][0] = words.low();
        current[r][1] = words.high();
    }

    for (int group = 0; group < groups; group++) {
        if (group + 1 == groups && last_pixels < 32) {  // the lanes beyond the row: zero padding
#pragma clang loop unroll(full)
            for (int r = 0; r < 3; r++) {
                current[r][0] = select(last_lanes, current[r][0], Vector<uint8_t>::zero());
                current[r][1] = select(last_lanes, current[r][1], Vector<uint8_t>::zero());
            }
        }

#pragma clang loop unroll(full)
        for (int r = 0; r < 3; r++) {
            if (group + 1 < groups) {
                const VectorPair<uint8_t> words =
                    deinterleave_chunks<4>(rows[r][2 * group + 3], rows[r][2 * group + 2]);
                next[r][0] = words.low();
                next[r][1] = words.high();
            } else {
                next[r][0] = Vector<uint8_t>::zero();
                next[r][1] = Vector<uint8_t>::zero();
            }
        }

        Vector<uint8_t> taps[9][2];
#pragma clang loop unroll(full)
        for (int r = 0; r < 3; r++) {
#pragma clang loop unroll(full)
            for (int c = 0; c < 2; c++) {
                taps[3 * r][c] = previous_pixels(current[r][c], previous[r][c]);
                taps[3 * r + 1][c] = current[r][c];
                taps[3 * r + 2][c] = next_pixels(current[r][c], next[r][c]);
            }
        }

        Vector<int32_t> acc[8];
#pragma clang loop unroll(full)
        for (int n = 0; n < 8; n++) {
            acc[n] = Vector<int32_t>::zero();
#pragma clang loop unroll(full)
            for (int t = 0; t < 9; t++) {
                acc[n] = dot_product_accumulate_scalar(acc[n], taps[t][0], weights[18 * n + 2 * t]);
                acc[n] = dot_product_accumulate_scalar(acc[n], taps[t][1], weights[18 * n + 2 * t + 1]);
            }
        }
        store_eight_channels(job, acc, output + 2 * group, job->output_width - 32 * group);

#pragma clang loop unroll(full)
        for (int r = 0; r < 3; r++) {
#pragma clang loop unroll(full)
            for (int c = 0; c < 2; c++) {
                previous[r][c] = current[r][c];
                current[r][c] = next[r][c];
            }
        }
    }
}

struct Layer3Job {
    const uint8_t * input;
    int input_width;
    const uint8_t * image;
    int image_row_bytes;
    uint8_t * output;
    int output_width;
    const int32_t * weights;
    Vector<float> scales[24];
    Vector<float> biases[24];
    Vector<float> skip_multipliers[24];
    Vector<float> skip_offsets[24];
    Vector<float> inverse_scales[24];
};

inline void store_block2_input(const Layer3Job * job, const Vector<int32_t> acc[24], HVX_Vector block_sums,
                               Vector<uint8_t> * output, int pixels) {
    const Vector<float> sums = to_float(Vector<int32_t>{block_sums});
    const Vector<float> half_offset = Vector<float>::splat(128.5f);
    Vector<int32_t> q[24];
#pragma clang loop unroll(full)
    for (int n = 0; n < 24; n++) {
        const Vector<float> x1 = max(to_float(acc[n]) * job->scales[n] + job->biases[n], Vector<float>::zero());
        const Vector<float> skip = sums * job->skip_multipliers[n] + job->skip_offsets[n];
        const Vector<float> t = x1 + skip;
        q[n] = truncate_to_int32(t * job->inverse_scales[n] + half_offset);
    }

    Vector<uint8_t> words[8];
#pragma clang loop unroll(full)
    for (int g = 0; g < 6; g++)
        words[g] = pack_to_uint8_interleaved(q[4 * g], q[4 * g + 1], q[4 * g + 2], q[4 * g + 3]);
    words[6] = reinterpret<uint8_t>(Vector<uint32_t>::splat_bits(0x80808080u));
    words[7] = words[6];

    VectorPair<uint8_t> pairs[4];
#pragma clang loop unroll(full)
    for (int p = 0; p < 4; p++) pairs[p] = interleave_chunks<4>(words[2 * p + 1], words[2 * p]);

    VectorPair<uint8_t> quads[4];
#pragma clang loop unroll(full)
    for (int half = 0; half < 2; half++) {
        quads[2 * half] =
            interleave_chunks<8>(half ? pairs[1].high() : pairs[1].low(), half ? pairs[0].high() : pairs[0].low());
        quads[2 * half + 1] =
            interleave_chunks<8>(half ? pairs[3].high() : pairs[3].low(), half ? pairs[2].high() : pairs[2].low());
    }

    Vector<uint8_t> vectors[8];
#pragma clang loop unroll(full)
    for (int half = 0; half < 2; half++) {
        const VectorPair<uint8_t> first = interleave_chunks<16>(quads[2 * half + 1].low(), quads[2 * half].low());
        const VectorPair<uint8_t> second = interleave_chunks<16>(quads[2 * half + 1].high(), quads[2 * half].high());
        vectors[4 * half] = first.low();
        vectors[4 * half + 1] = first.high();
        vectors[4 * half + 2] = second.low();
        vectors[4 * half + 3] = second.high();
    }

    if (pixels >= 32) {
#pragma clang loop unroll(full)
        for (int v = 0; v < 8; v++) output[v] = vectors[v];
    } else {
        for (int v = 0; v < pixels / 4; v++) output[v] = vectors[v];
    }
}

void layer3_row(void * context, int y, int) {
    const auto * job = static_cast<const Layer3Job *>(context);
    const int32_t * weights = job->weights;
    const Vector<uint8_t> * rows[3];
    for (int r = 0; r < 3; r++) {
        rows[r] = hvx::vectors(job->input) +
                  static_cast<ptrdiff_t>(2 * y - 1 + r) * static_cast<ptrdiff_t>(job->input_width / 16);
    }
    const uint8_t * image_rows = job->image + static_cast<size_t>(4 * y) * job->image_row_bytes;
    auto * output = hvx::vectors(job->output + static_cast<size_t>(y) * job->output_width * 32);
    Vector<uint8_t> previous_odd[3][2] = {};

    for (int group = 0; group * 32 < job->output_width; group++) {
        Vector<uint8_t> taps[9][2];
#pragma clang loop unroll(full)
        for (int r = 0; r < 3; r++) {
            const Vector<uint8_t> * words = rows[r] + 4 * group;
            const VectorPair<uint8_t> first = deinterleave_chunks<4>(words[1], words[0]);
            const VectorPair<uint8_t> second = deinterleave_chunks<4>(words[3], words[2]);
#pragma clang loop unroll(full)
            for (int c = 0; c < 2; c++) {
                const VectorPair<uint8_t> pixels =
                    deinterleave_chunks<4>(c ? second.high() : second.low(), c ? first.high() : first.low());
                const Vector<uint8_t> even = pixels.low();
                const Vector<uint8_t> odd = pixels.high();
                taps[3 * r][c] = previous_pixels(odd, previous_odd[r][c]);
                taps[3 * r + 1][c] = even;
                taps[3 * r + 2][c] = odd;
                previous_odd[r][c] = odd;
            }
        }

        Vector<int32_t> acc[24];
#pragma clang loop unroll(full)
        for (int n = 0; n < 24; n++) {
            acc[n] = Vector<int32_t>::zero();
#pragma clang loop unroll(full)
            for (int t = 0; t < 9; t++) {
                acc[n] = dot_product_accumulate_scalar(acc[n], taps[t][0], weights[18 * n + 2 * t]);
                acc[n] = dot_product_accumulate_scalar(acc[n], taps[t][1], weights[18 * n + 2 * t + 1]);
            }
        }

        Vector<int32_t> block_sums = Vector<int32_t>::zero();
#pragma clang loop unroll(full)
        for (int r = 0; r < 4; r++) {
            block_sums = dot_product_accumulate_scalar(
                block_sums, hvx::load(image_rows + static_cast<size_t>(r) * job->image_row_bytes + 128 * group), kOnes);
        }
        store_block2_input(job, acc, block_sums.bits, output + 8 * group, job->output_width - 32 * group);
    }
}

runtime::Result<ImageNormalization> normalize_image(runtime::Session & session, const uint8_t * image, int height,
                                                    int width) {
    SumJob job{image, block1_padded_image_row_bytes(width), height, {}, {}};
    const runtime::Status status = runtime::run_parallel(session, runtime::kHvxThreads, sum_band, &job);
    if (!status) return std::unexpected(status.error());

    uint64_t s1 = 0;
    uint64_t s2 = 0;
    for (int thread = 0; thread < runtime::kHvxThreads; thread++) {
        s1 += job.sums[thread];
        s2 += job.squares[thread];
    }

    const auto n = static_cast<int64_t>(height) * width;
    const int64_t numerator = n * static_cast<int64_t>(s2) - static_cast<int64_t>(s1) * static_cast<int64_t>(s1);
    const double variance = static_cast<double>(numerator) / static_cast<double>(n * n);
    ImageNormalization normalization;
    normalization.mean = static_cast<float>(static_cast<double>(s1) / static_cast<double>(n));
    normalization.inverse_std = static_cast<float>(1.0 / sqrt(variance + 1e-5));
    return normalization;
}

runtime::Status run_block1(runtime::Session & session, const Block1Arguments & arguments) {
    const int image_row_bytes = block1_padded_image_row_bytes(arguments.width);

    // The normalization folded into layer 0: m = k inv_std, b = beta - (mean S) m, S the border class's tap sum.
    {
        const LayerTensor tensor = layer_tensor(arguments.tensors[0], 12);
        const auto * tap_sums = reinterpret_cast<const int32_t *>(tensor.constants + 64);
        Layer0Job job{arguments.image, image_row_bytes, arguments.height, arguments.width, arguments.map0, {}, {}, {}};
        memcpy(job.weights, tensor.weights, sizeof job.weights);
        for (int n = 0; n < 4; n++) {
            const float multiplier = tensor.constants[n] * arguments.normalization.inverse_std;
            job.multipliers[n] = multiplier;
            for (int border_class = 0; border_class < 9; border_class++) {
                const float shift = arguments.normalization.mean * static_cast<float>(tap_sums[32 * border_class + n]);
                const float scaled_shift = shift * multiplier;
                job.biases[n][border_class] = tensor.constants[32 + n] - scaled_shift;
            }
        }
        const runtime::Status status =
            runtime::parallel_for(session, runtime::kHvxThreads, arguments.height, layer0_row, &job);
        if (!status) return status;
    }

    {
        const LayerTensor tensor = layer_tensor(arguments.tensors[1], 72);
        EightChannelJob job{
            arguments.map0, arguments.width, arguments.map1, arguments.width / 2, tensor.weights, {}, {}};
        for (int n = 0; n < 8; n++) {
            job.multipliers[n] = Vector<float>::splat(tensor.constants[n]);
            job.biases[n] = Vector<float>::splat(tensor.constants[32 + n]);
        }
        const runtime::Status status =
            runtime::parallel_for(session, runtime::kHvxThreads, arguments.height / 2, layer1_row, &job);
        if (!status) return status;
    }

    {
        const LayerTensor tensor = layer_tensor(arguments.tensors[2], 144);
        EightChannelJob job{
            arguments.map1, arguments.width / 2, arguments.map2, arguments.width / 2, tensor.weights, {}, {}};
        for (int n = 0; n < 8; n++) {
            job.multipliers[n] = Vector<float>::splat(tensor.constants[n]);
            job.biases[n] = Vector<float>::splat(tensor.constants[32 + n]);
        }
        const runtime::Status status =
            runtime::parallel_for(session, runtime::kHvxThreads, arguments.height / 2, layer2_row, &job);
        if (!status) return status;
    }

    // skip = Ws avg_pool4((x - mean) inv_std) + bs = block sum u + v, u = Ws / 16 inv_std, v = bs - (Ws mean) inv_std
    {
        const LayerTensor tensor = layer_tensor(arguments.tensors[3], 432);
        Layer3Job job{arguments.map2,
                      arguments.width / 2,
                      arguments.image,
                      image_row_bytes,
                      arguments.output,
                      arguments.width / 4,
                      tensor.weights,
                      {},
                      {},
                      {},
                      {},
                      {}};
        for (int n = 0; n < 24; n++) {
            job.scales[n] = Vector<float>::splat(tensor.constants[n]);
            job.biases[n] = Vector<float>::splat(tensor.constants[32 + n]);
            const float multiplier = tensor.constants[64 + n] * arguments.normalization.inverse_std;
            const float weighted_mean = tensor.constants[96 + n] * arguments.normalization.mean;
            const float scaled_mean = weighted_mean * arguments.normalization.inverse_std;
            job.skip_multipliers[n] = Vector<float>::splat(multiplier);
            job.skip_offsets[n] = Vector<float>::splat(tensor.constants[128 + n] - scaled_mean);
            job.inverse_scales[n] = Vector<float>::splat(tensor.constants[160 + n]);
        }
        const runtime::Status status =
            runtime::parallel_for(session, runtime::kHvxThreads, arguments.height / 4, layer3_row, &job);
        if (!status) return status;
    }
    return {};
}

// Cell (Y, X)'s byte 8 ky + kx is pixel (8 Y + ky, 8 X + kx), unfold2d's channel order.
struct SpaceToDepthJob {
    const uint8_t * image;
    size_t row_bytes;
    int cell_columns;
    uint8_t * cells;
};

void space_to_depth_row(void * context, int cell_y, int) {
    const auto * job = static_cast<const SpaceToDepthJob *>(context);
    const uint8_t * rows = job->image + static_cast<size_t>(8 * cell_y) * job->row_bytes;
    auto * output = hvx::vectors(job->cells + static_cast<size_t>(cell_y) * job->cell_columns * 64);

    for (int chunk = 0; chunk * 16 < job->cell_columns; chunk++) {
        Vector<uint8_t> r[8];
        for (int k = 0; k < 8; k++) r[k] = hvx::load(rows + k * job->row_bytes + 128 * chunk);

        VectorPair<uint8_t> pairs[4];
        for (int k = 0; k < 4; k++) pairs[k] = interleave_chunks<8>(r[2 * k + 1], r[2 * k]);

        VectorPair<uint8_t> quads[4];
        for (int half = 0; half < 2; half++) {
            quads[2 * half] =
                interleave_chunks<16>(half ? pairs[1].high() : pairs[1].low(), half ? pairs[0].high() : pairs[0].low());
            quads[2 * half + 1] =
                interleave_chunks<16>(half ? pairs[3].high() : pairs[3].low(), half ? pairs[2].high() : pairs[2].low());
        }

        const int cells = job->cell_columns - 16 * chunk < 16 ? job->cell_columns - 16 * chunk : 16;
        Vector<uint8_t> out[8];
        for (int half = 0; half < 2; half++) {
            const VectorPair<uint8_t> first = interleave_chunks<32>(quads[2 * half + 1].low(), quads[2 * half].low());
            const VectorPair<uint8_t> second =
                interleave_chunks<32>(quads[2 * half + 1].high(), quads[2 * half].high());
            out[4 * half] = first.low();
            out[4 * half + 1] = first.high();
            out[4 * half + 2] = second.low();
            out[4 * half + 3] = second.high();
        }
        for (int v = 0; v < (cells + 1) / 2; v++) output[8 * chunk + v] = out[v];
    }
}

// up2 / up4 (bilinear, of the codes) are exact in fp32: the weights are multiples of 1/8, the codes bytes.
struct FusionJob {
    const uint8_t * x3;
    const uint8_t * x4;
    const uint8_t * x5;
    uint8_t * output;
    int height;
    int width;
    Vector<float> scales[3][2];
};

// PyTorch's bilinear source index and weights (area_pixel_compute_source_index, align_corners False).
struct UpsampleTap {
    int first;
    int second;
    float first_weight;
    float second_weight;
};

UpsampleTap upsample_tap(int i, int factor, int size) {
    const float source = (static_cast<float>(i) + 0.5f) / static_cast<float>(factor) - 0.5f;
    const float clamped = source < 0.0f ? 0.0f : source;
    const int first = static_cast<int>(clamped);
    const float lambda = clamped - static_cast<float>(first);
    return {first, first < size - 1 ? first + 1 : first, 1.0f - lambda, lambda};
}

inline void channels_f32(const uint8_t * pixel, Vector<float> * out) {
    const VectorPair<uint16_t> halfwords = widen(hvx::load_unaligned(pixel));
    const VectorPair<uint32_t> words = widen(halfwords.low());
    out[0] = to_float(reinterpret<int32_t>(words.low()));
    out[1] = to_float(reinterpret<int32_t>(words.high()));
}

inline void bilinear_channels(const uint8_t * map, int width, const UpsampleTap & row, const UpsampleTap & column,
                              Vector<float> * out) {
    Vector<float> a[2];
    Vector<float> b[2];
    Vector<float> c[2];
    Vector<float> d[2];
    channels_f32(map + (static_cast<size_t>(row.first) * width + column.first) * 64, a);
    channels_f32(map + (static_cast<size_t>(row.first) * width + column.second) * 64, b);
    channels_f32(map + (static_cast<size_t>(row.second) * width + column.first) * 64, c);
    channels_f32(map + (static_cast<size_t>(row.second) * width + column.second) * 64, d);

    const Vector<float> w0 = Vector<float>::splat(column.first_weight);
    const Vector<float> w1 = Vector<float>::splat(column.second_weight);
    const Vector<float> h0 = Vector<float>::splat(row.first_weight);
    const Vector<float> h1 = Vector<float>::splat(row.second_weight);
    for (int k = 0; k < 2; k++) {
        const Vector<float> top = w0 * a[k] + w1 * b[k];
        const Vector<float> bottom = w0 * c[k] + w1 * d[k];
        out[k] = h0 * top + h1 * bottom;
    }
}

void fusion_row(void * context, int y, int) {
    const auto * job = static_cast<const FusionJob *>(context);
    const int width = job->width;
    const UpsampleTap row4 = upsample_tap(y, 2, job->height / 2);
    const UpsampleTap row5 = upsample_tap(y, 4, job->height / 4);
    const Vector<float> half = Vector<float>::splat(0.5f);

    for (int x = 0; x < width; x += 2) {
        Vector<int32_t> q[2][2];
        for (int p = 0; p < 2; p++) {
            Vector<float> x3[2];
            Vector<float> u4[2];
            Vector<float> u5[2];
            channels_f32(job->x3 + (static_cast<size_t>(y) * width + x + p) * 64, x3);
            bilinear_channels(job->x4, width / 2, row4, upsample_tap(x + p, 2, width / 2), u4);
            bilinear_channels(job->x5, width / 4, row5, upsample_tap(x + p, 4, width / 4), u5);
            for (int k = 0; k < 2; k++) {
                const Vector<float> a = x3[k] * job->scales[0][k];
                const Vector<float> b = u4[k] * job->scales[1][k];
                const Vector<float> c = u5[k] * job->scales[2][k];
                const Vector<float> v = a + b + c + half;
                q[p][k] = truncate_to_int32(v);
            }
        }

        const Vector<int16_t> first = pack_to_int16_saturating(q[0][1], q[0][0]);
        const Vector<int16_t> second = pack_to_int16_saturating(q[1][1], q[1][0]);
        store(job->output + (static_cast<size_t>(y) * width + x) * 64, pack_to_uint8_saturating(second, first));
    }
}

struct HeatmapInputJob {
    const float * descriptor_map;
    uint8_t * output;
    int cells;
    Vector<float> inverse_scales[2];
};

void quantize_heatmap_input(void * context, int item, int) {
    const auto * job = static_cast<const HeatmapInputJob *>(context);
    const Vector<float> offset = Vector<float>::splat(128.5f);
    const int first = kHeatmapInputItemCells * item;

    for (int cell = first; cell < first + kHeatmapInputItemCells && cell < job->cells; cell += 2) {
        Vector<int32_t> q[2][2];
        for (int p = 0; p < 2; p++) {
            const auto * row = hvx::vectors(job->descriptor_map + static_cast<size_t>(cell + p) * 64);
            for (int k = 0; k < 2; k++) q[p][k] = truncate_to_int32(row[k] * job->inverse_scales[k] + offset);
        }
        const Vector<int16_t> first = pack_to_int16_saturating(q[0][1], q[0][0]);
        const Vector<int16_t> second = pack_to_int16_saturating(q[1][1], q[1][0]);
        store(job->output + static_cast<size_t>(cell) * 64, pack_to_uint8_saturating(second, first));
    }
}

struct ReliabilityJob {
    const float * rows;
    float * reliability;
    int cells;
};

void compute_reliability(void * context, int item, int) {
    const auto * job = static_cast<const ReliabilityJob *>(context);
    const int first = kReliabilityItemCells * item;
    const int cells = job->cells - first < kReliabilityItemCells ? job->cells - first : kReliabilityItemCells;
    hvx::prefetch_to_l2(job->rows + static_cast<size_t>(first) * kReliabilityColumns, 4 * kReliabilityColumns,
                        4 * kReliabilityColumns, cells);

    for (int cell = first; cell < first + cells; cell++) {
        const float z = job->rows[static_cast<size_t>(cell) * kReliabilityColumns];
        // sigmoid; this expf returns a wrong finite value for arguments > 88, where 1 / (1 + e) is 0
        job->reliability[cell] = -z > 88.0f ? 0.0f : 1.0f / (1.0f + expf(-z));
    }
}

struct CandidateJob {
    const float * heatmap;
    int height;
    int width;
    int stride;
    float threshold;
    const float * reliability;
    float * rings[runtime::kHvxThreads];
    float *
        row_buffers[runtime::kHvxThreads];  // per thread: the row in cached memory (scalar loads from VTCM are slow)
    Workspace * workspace;
};

// The max over x - 2 .. x + 2, zeros beyond the row (= max_pool2d's -inf padding: the values are >= 0).
void horizontal_maxima(const Vector<float> * row, Vector<float> * out, int vectors) {
    const Vector<float> zero = Vector<float>::zero();
    for (int v = 0; v < vectors; v++) {
        const Vector<float> previous = v > 0 ? row[v - 1] : zero;
        const Vector<float> current = row[v];
        const Vector<float> next = v + 1 < vectors ? row[v + 1] : zero;
        Vector<float> m = max(current, slide_up(current, previous, 4));
        m = max(m, slide_up(current, previous, 8));
        m = max(m, slide_down(next, current, 4));
        out[v] = max(m, slide_down(next, current, 8));
    }
}

// As detectAndCompute: nearest(heatmap) x bilinear(reliability), grid_sample's formulas with zeros outside.
float keypoint_score(const CandidateJob * job, int x, int y, const float * row_values) {
    const SampleTaps & column = job->workspace->column_taps[x];
    const SampleTaps & row = job->workspace->row_taps[y];

    const bool inside =
        column.nearest >= 0 && column.nearest < job->width && row.nearest >= 0 && row.nearest < job->height;
    const float nearest = !inside ? 0.0f
                          : row.nearest == y
                              ? row_values[column.nearest]
                              : job->heatmap[static_cast<size_t>(row.nearest) * job->stride + column.nearest];

    const int cell_width = job->width / 8;
    const int cell_height = job->height / 8;
    const auto value = [&](int cx, int cy) {
        return cx >= 0 && cx < cell_width && cy >= 0 && cy < cell_height
                   ? job->reliability[static_cast<size_t>(cy) * cell_width + cx]
                   : 0.0f;
    };
    const float nw = column.first_weight * row.first_weight;
    const float ne = column.second_weight * row.first_weight;
    const float sw = column.first_weight * row.second_weight;
    const float se = column.second_weight * row.second_weight;
    float bilinear = value(column.first, row.first) * nw;
    const float t1 = value(column.first + 1, row.first) * ne;
    bilinear = bilinear + t1;
    const float t2 = value(column.first, row.first + 1) * sw;
    bilinear = bilinear + t2;
    const float t3 = value(column.first + 1, row.first + 1) * se;
    bilinear = bilinear + t3;
    return nearest * bilinear;
}

// Scores <= 0 are dropped as in detectAndCompute, where (0, 0) is a padding marker of score -1.
void collect_candidates(void * job_pointer, int thread) {
    const auto * job = static_cast<const CandidateJob *>(job_pointer);
    Workspace & workspace = *job->workspace;
    const int height = job->height;
    const int vectors = job->stride / 32;
    const int first_y = height * thread / runtime::kHvxThreads;
    const int end_y = height * (thread + 1) / runtime::kHvxThreads;
    const Vector<float> threshold = Vector<float>::splat(job->threshold);
    const Vector<uint32_t> lane_bits = hvx::load(hvx::kWordLaneBits.bits);
    float * ring = job->rings[thread];
    const auto ring_row = [&](int y) { return hvx::vectors(ring + static_cast<size_t>((y + 5) % 5) * job->stride); };
    const auto heatmap_row = [&](int y) { return hvx::vectors(job->heatmap + static_cast<size_t>(y) * job->stride); };
    float * candidate_scores = workspace.candidate_scores[thread].get();
    int32_t * candidate_positions = workspace.candidate_positions[thread].get();
    int count = 0;

    for (int y = first_y - 2; y < first_y + 2; y++) {
        if (y >= 0 && y < height) horizontal_maxima(heatmap_row(y), ring_row(y), vectors);
    }

    alignas(128) uint32_t row_masks[kMaximumSide / 32];
    float * row_buffer = job->row_buffers[thread];
    auto * row_values = hvx::vectors(row_buffer);
    for (int y = first_y; y < end_y; y++) {
        if (y + 2 < height) horizontal_maxima(heatmap_row(y + 2), ring_row(y + 2), vectors);

        const Vector<float> * row = heatmap_row(y);
        Vector<uint32_t> masks = Vector<uint32_t>::zero();
        for (int v = 0; v < vectors; v++) {
            Vector<float> maximum = Vector<float>::zero();
            for (int dy = -2; dy <= 2; dy++) {
                if (y + dy >= 0 && y + dy < height) maximum = max(maximum, ring_row(y + dy)[v]);
            }
            const Mask local_maximum =
                (reinterpret<int32_t>(row[v]) == reinterpret<int32_t>(maximum)) & (row[v] > threshold);
            Vector<uint32_t> bits = select(local_maximum, lane_bits, Vector<uint32_t>::zero());
            bits = vector_math::or_to_all_lanes(bits);
            masks = slide_down(bits, masks, 4);
            row_values[v] = row[v];
            if (v % 32 == 31 || v == vectors - 1) {
                store(row_masks + v / 32 * 32, slide_down(Vector<uint32_t>::zero(), masks, 4 * (31 - v % 32)));
            }
        }

        for (int v = 0; v < vectors; v++) {
            for (uint32_t mask = row_masks[v]; mask != 0; mask &= mask - 1) {
                const int x = 32 * v + static_cast<int>(Q6_R_ct0_R(mask));
                const float score = (x == 0 && y == 0) ? -1.0f : keypoint_score(job, x, y, row_buffer);
                if (score > 0.0f && count < workspace.candidate_capacity) {
                    candidate_scores[count] = score;
                    candidate_positions[count++] = (y << 16) | x;
                }
            }
        }
    }

    workspace.candidate_counts[thread] = count;
}

// Stable LSD radix sort by descending score on ~bits: the bit patterns of floats > 0 order as their values.
void sort_band(void * job_pointer, int thread) {
    Workspace & workspace = *static_cast<Workspace *>(job_pointer);
    const int count = workspace.candidate_counts[thread];
    float * scores = workspace.candidate_scores[thread].get();
    int32_t * positions = workspace.candidate_positions[thread].get();
    float * other_scores = workspace.candidate_scratch_scores[thread].get();
    int32_t * other_positions = workspace.candidate_scratch_positions[thread].get();

    int histograms[4][256] = {};
    for (int c = 0; c < count; c++) {
        const uint32_t key = ~std::bit_cast<uint32_t>(scores[c]);
        histograms[0][key & 255]++;
        histograms[1][(key >> 8) & 255]++;
        histograms[2][(key >> 16) & 255]++;
        histograms[3][key >> 24]++;
    }

    for (int digit = 0; digit < 4 && count > 1; digit++) {
        const int shift = 8 * digit;
        int * histogram = histograms[digit];
        if (histogram[(~std::bit_cast<uint32_t>(scores[0]) >> shift) & 255] == count) continue;

        int offset = 0;
        for (int d = 0; d < 256; d++) {
            const int n = histogram[d];
            histogram[d] = offset;
            offset += n;
        }
        for (int c = 0; c < count; c++) {
            const int slot = histogram[(~std::bit_cast<uint32_t>(scores[c]) >> shift) & 255]++;
            other_scores[slot] = scores[c];
            other_positions[slot] = positions[c];
        }
        std::swap(scores, other_scores);
        std::swap(positions, other_positions);
    }

    workspace.sorted_scores[thread] = scores;
    workspace.sorted_positions[thread] = positions;
}

struct DescriptorJob {
    float * descriptor_map;
    int cells;
    int cell_width;
    int cell_height;
    const float * keypoints;
    int keypoint_count;
    _Float16 * descriptors;
    const Workspace * workspace;
};

inline Vector<float> inverse_norm(Vector<float> sum) {
    alignas(128) float lanes[32];
    store(lanes, sum);
    const float norm = sqrtf(lanes[0]);
    return Vector<float>::splat(1.0f / (norm > 1e-12f ? norm : 1e-12f));
}

void normalize_cells(void * context, int item, int) {
    const auto * job = static_cast<const DescriptorJob *>(context);
    const int count = job->cells - 32 * item < 32 ? job->cells - 32 * item : 32;
    auto * rows = hvx::vectors(job->descriptor_map + static_cast<size_t>(32 * item) * 64);

    Vector<float> sums = Vector<float>::zero();
    for (int c = 0; c < 32; c++) {
        const int cell = c < count ? c : 0;
        const Vector<float> sum =
            vector_math::sum_to_all_lanes(rows[2 * cell] * rows[2 * cell] + rows[2 * cell + 1] * rows[2 * cell + 1]);
        sums = slide_down(sum, sums, 4);  // lane i: cell i after the 32
    }

    // 1 / max(norm, 1e-12), F.normalize's eps
    const Vector<float> reciprocal = min(vector_math::precise_reciprocal_square_root(sums), 1e12f);
    alignas(128) float scales[32];
    store(scales, reciprocal);

    for (int c = 0; c < count; c++) {
        const Vector<float> scale = Vector<float>::splat(scales[c]);
        rows[2 * c] = rows[2 * c] * scale;
        rows[2 * c + 1] = rows[2 * c + 1] * scale;
    }
}

// PyTorch's cubic convolution coefficients (A = -0.75) of fraction t.
inline void cubic_coefficients(float t, float coefficients[4]) {
    constexpr float A = -0.75f;
    const auto inner = [](float x) { return ((A + 2.0f) * x - (A + 3.0f)) * x * x + 1.0f; };
    const auto outer = [](float x) { return ((A * x - 5.0f * A) * x + 8.0f * A) * x - 4.0f * A; };
    coefficients[0] = outer(t + 1.0f);
    coefficients[1] = inner(t);
    coefficients[2] = inner(1.0f - t);
    coefficients[3] = outer(1.0f - t + 1.0f);
}

void sample_descriptors(void * context, int item, int) {
    const auto * job = static_cast<const DescriptorJob *>(context);
    const int first = kSampleItemKeypoints * item;

    for (int k = first; k < first + kSampleItemKeypoints && k < job->keypoint_count; k++) {
        const SampleTaps & column = job->workspace->column_taps[static_cast<int>(job->keypoints[2 * k])];
        const SampleTaps & row = job->workspace->row_taps[static_cast<int>(job->keypoints[2 * k + 1])];
        float column_weights[4];
        float row_weights[4];
        cubic_coefficients(column.second_weight, column_weights);
        cubic_coefficients(row.second_weight, row_weights);
        const int x0 = column.first - 1;
        const int y0 = row.first - 1;

        Vector<float> sample[2] = {Vector<float>::zero(), Vector<float>::zero()};
        for (int r = 0; r < 4; r++) {
            Vector<float> row_sum[2] = {Vector<float>::zero(), Vector<float>::zero()};
            for (int c = 0; c < 4; c++) {
                const int cx = x0 + c;
                const int cy = y0 + r;
                if (cx < 0 || cx >= job->cell_width || cy < 0 || cy >= job->cell_height) continue;
                const auto * cell =
                    hvx::vectors(job->descriptor_map + (static_cast<size_t>(cy) * job->cell_width + cx) * 64);
                const Vector<float> weight = Vector<float>::splat(column_weights[c]);
                row_sum[0] = row_sum[0] + cell[0] * weight;
                row_sum[1] = row_sum[1] + cell[1] * weight;
            }
            const Vector<float> weight = Vector<float>::splat(row_weights[r]);
            sample[0] = sample[0] + row_sum[0] * weight;
            sample[1] = sample[1] + row_sum[1] * weight;
        }

        const Vector<float> scale =
            inverse_norm(vector_math::sum_to_all_lanes(sample[0] * sample[0] + sample[1] * sample[1]));
        const VectorPair<float> dealt = deinterleave(sample[1] * scale, sample[0] * scale);
        store_unaligned(job->descriptors + static_cast<size_t>(k) * 64,
                        to_float16_interleaved(dealt.low(), dealt.high()));
    }
}

struct Convolution {
    int tensor;
    int height;
    int width;
    int channels;
    int taps;
    int stride;
    int zero_point;
    int N;
    const uint8_t * input;
    uint8_t * output;
    float * rows_f32;
};

runtime::Status run_convolution(runtime::Session & session, const LayerWeights & model, const Convolution & c,
                                const float * multipliers = nullptr, const float * biases = nullptr) {
    convolution_int8::Arguments arguments;
    arguments.input = c.input;
    arguments.height = c.height;
    arguments.width = c.width;
    arguments.channels = c.channels;
    arguments.taps = c.taps;
    arguments.stride = c.stride;
    arguments.zero_point = c.zero_point;
    arguments.weights = model.part<uint8_t>(c.tensor);
    arguments.N = c.N;

    if (c.output != nullptr) {
        arguments.uint8_outputs[0] = c.output;
        arguments.uint8_output_channels = c.N;
        arguments.output_multipliers = multipliers;
        arguments.output_biases = biases;
    } else {
        arguments.output_form = convolution_int8::OutputForm::kRowsF32;
        arguments.rows_f32 = c.rows_f32;
    }

    return kernels::convolution_int8::run(session, arguments);
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
    if (!session.model_mapped(model_slot) || height % 32 != 0 || width % 32 != 0 || height < 64 || width < 64 ||
        height > kMaximumSide || width > kMaximumSide || max_keypoints < 1 || features == nullptr) {
        return std::unexpected(runtime::Error::kBadParameter);
    }

    runtime::Status status = reserve_features(features, max_keypoints, protocol::xfeat::kDescriptorDimensions);
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

    const size_t image_row_bytes = padded_image_row_bytes(width);
    uint8_t * const image_rows = workspace.image.get() + image_row_bytes;
    runtime::invalidate_host_input(arguments.image, static_cast<size_t>(height) * width);
    ImageCopyJob copy_job{arguments.image, image_rows, image_row_bytes, height, width};
    status = runtime::run_parallel(session, runtime::kHvxThreads, copy_image_band, &copy_job);
    if (!status) return status;

    const runtime::Result<ImageNormalization> normalization = normalize_image(session, image_rows, height, width);
    if (!normalization) return std::unexpected(normalization.error());
    statistics[stat::kImageMeanBits] = std::bit_cast<int32_t>(normalization->mean);
    statistics[stat::kImageInverseStdBits] = std::bit_cast<int32_t>(normalization->inverse_std);

    const size_t map0_row = static_cast<size_t>(width) * 4;
    const size_t map1_row = static_cast<size_t>(width / 2) * 8;
    const size_t map0_bytes = (static_cast<size_t>(height) + 2) * map0_row;
    const size_t map1_bytes = (static_cast<size_t>(height / 2) + 2) * map1_row;
    if (map0_bytes + 2 * map1_bytes > session.vtcm_bytes()) return std::unexpected(runtime::Error::kNoMemory);
    uint8_t * vtcm = session.vtcm();
    uint8_t * map0 = vtcm + map0_row;
    uint8_t * map1 = vtcm + map0_bytes + map1_row;
    uint8_t * map2 = vtcm + map0_bytes + map1_bytes + map1_row;

    for (size_t v = 0; v < map0_row / kVectorBytes; v++) {  // the zero rows by vector stores (VTCM)
        hvx::vectors(map0 - map0_row)[v] = Vector<uint8_t>::zero();
        hvx::vectors(map0 + height * map0_row)[v] = Vector<uint8_t>::zero();
    }
    for (size_t v = 0; v < map1_row / kVectorBytes; v++) {
        hvx::vectors(map1 - map1_row)[v] = Vector<uint8_t>::zero();
        hvx::vectors(map1 + (height / 2) * map1_row)[v] = Vector<uint8_t>::zero();
        hvx::vectors(map2 - map1_row)[v] = Vector<uint8_t>::zero();
        hvx::vectors(map2 + (height / 2) * map1_row)[v] = Vector<uint8_t>::zero();
    }

    Block1Arguments block1{image_rows,
                           height,
                           width,
                           *normalization,
                           {model.part<uint8_t>(tensor::kBlock1Layer0), model.part<uint8_t>(tensor::kBlock1Layer1),
                            model.part<uint8_t>(tensor::kBlock1Layer2), model.part<uint8_t>(tensor::kBlock1Layer3)},
                           map0,
                           map1,
                           map2,
                           workspace.block2_input.get()};
    status = run_block1(session, block1);
    if (!status) return status;
    timer.lap(stat::kBlock1Microseconds);

    uint8_t * maps[kMapCount];
    for (int map = 0; map < kMapCount; map++) maps[map] = workspace.maps[map].get();
    const int h4 = height / 4;
    const int w4 = width / 4;
    const int h8 = height / 8;
    const int w8 = width / 8;
    const int h16 = height / 16;
    const int w16 = width / 16;

    const Convolution backbone[] = {
        {tensor::kBlock2Layer0, h4, w4, 32, 9, 1, 128, 32, workspace.block2_input.get(), maps[kBlock2First], nullptr},
        {tensor::kBlock2Layer1, h4, w4, 32, 9, 1, 0, 32, maps[kBlock2First], maps[kBlock2Second], nullptr},
        {tensor::kBlock3Layer0, h4, w4, 32, 9, 2, 0, 64, maps[kBlock2Second], maps[kBlock3First], nullptr},
        {tensor::kBlock3Layer1, h8, w8, 64, 9, 1, 0, 64, maps[kBlock3First], maps[kBlock3Second], nullptr},
        {tensor::kBlock3Layer2, h8, w8, 64, 1, 1, 0, 64, maps[kBlock3Second], maps[kX3], nullptr},
        {tensor::kBlock4Layer0, h8, w8, 64, 9, 2, 0, 64, maps[kX3], maps[kBlock4First], nullptr},
        {tensor::kBlock4Layer1, h16, w16, 64, 9, 1, 0, 64, maps[kBlock4First], maps[kBlock4Second], nullptr},
        {tensor::kBlock4Layer2, h16, w16, 64, 9, 1, 0, 64, maps[kBlock4Second], maps[kX4], nullptr},
        {tensor::kBlock5Layer0, h16, w16, 64, 9, 2, 0, 128, maps[kX4], maps[kBlock5First], nullptr},
        {tensor::kBlock5Layer1, height / 32, width / 32, 128, 9, 1, 0, 128, maps[kBlock5First], maps[kBlock5Second],
         nullptr},
        {tensor::kBlock5Layer2, height / 32, width / 32, 128, 9, 1, 0, 128, maps[kBlock5Second], maps[kBlock5Third],
         nullptr},
        {tensor::kBlock5Layer3, height / 32, width / 32, 128, 1, 1, 0, 64, maps[kBlock5Third], maps[kX5], nullptr},
    };
    for (const Convolution & c : backbone) {
        status = run_convolution(session, model, c);
        if (!status) return status;
    }
    timer.lap(stat::kBackboneMicroseconds);

    const auto * fusion_scales = model.part<float>(tensor::kFusionScales);
    FusionJob fusion_job{maps[kX3], maps[kX4], maps[kX5], maps[kFusionInput], h8, w8, {}};
    for (int s = 0; s < 3; s++) {
        for (int k = 0; k < 2; k++) fusion_job.scales[s][k] = hvx::load_unaligned(fusion_scales + 64 * s + 32 * k);
    }
    status = runtime::parallel_for(session, runtime::kHvxThreads, h8, fusion_row, &fusion_job);
    if (!status) return status;

    const Convolution fusion[] = {
        {tensor::kFusionLayer0, h8, w8, 64, 9, 1, 0, 64, maps[kFusionInput], maps[kBlock3First], nullptr},
        {tensor::kFusionLayer1, h8, w8, 64, 9, 1, 0, 64, maps[kBlock3First], maps[kBlock3Second], nullptr},
        {tensor::kFusionLayer2, h8, w8, 64, 1, 1, 0, 64, maps[kBlock3Second], nullptr, workspace.descriptor_map.get()},
    };
    for (const Convolution & c : fusion) {
        status = run_convolution(session, model, c);
        if (!status) return status;
    }
    timer.lap(stat::kFusionMicroseconds);

    const int cells = h8 * w8;
    HeatmapInputJob heatmap_input_job{workspace.descriptor_map.get(), maps[kHeatmapInput], cells, {}};
    for (int k = 0; k < 2; k++) heatmap_input_job.inverse_scales[k] = hvx::load_unaligned(fusion_scales + 192 + 32 * k);
    status = runtime::parallel_for(session, runtime::kHvxThreads,
                                   (cells + kHeatmapInputItemCells - 1) / kHeatmapInputItemCells,
                                   quantize_heatmap_input, &heatmap_input_job);
    if (!status) return status;
    SpaceToDepthJob space_to_depth_job{image_rows, image_row_bytes, w8, maps[kKeypointInput]};
    status = runtime::parallel_for(session, runtime::kHvxThreads, h8, space_to_depth_row, &space_to_depth_job);
    if (!status) return status;

    // The normalization folded into keypoint.0: m = k inv_std, b = beta - (mean S) m, S the channel's weight sum.
    {
        const uint8_t * keypoint_tensor = model.part<uint8_t>(tensor::kKeypointLayer0);
        const auto * constants =
            reinterpret_cast<const float *>(keypoint_tensor + kernels::convolution_int8::weight_stream_bytes(64, 64));
        const auto * tap_sums = reinterpret_cast<const int32_t *>(constants + 128);
        for (int n = 0; n < 64; n++) {
            const float multiplier = constants[n] * normalization->inverse_std;
            const float shift = normalization->mean * static_cast<float>(tap_sums[n]);
            const float scaled_shift = shift * multiplier;
            workspace.keypoint_multipliers[n] = multiplier;
            workspace.keypoint_biases[n] = constants[64 + n] - scaled_shift;
        }
    }

    const Convolution heads[] = {
        {tensor::kHeatmapLayer0, h8, w8, 64, 1, 1, 128, 64, maps[kHeatmapInput], maps[kHeadScratch], nullptr},
        {tensor::kHeatmapLayer1, h8, w8, 64, 1, 1, 0, 64, maps[kHeadScratch], maps[kFusionInput], nullptr},
        {tensor::kHeatmapLayer2, h8, w8, 64, 1, 1, 0, 32, maps[kFusionInput], nullptr,
         workspace.reliability_rows.get()},
        {tensor::kKeypointLayer0, h8, w8, 64, 1, 1, 0, 64, maps[kKeypointInput], maps[kHeadScratch], nullptr},
        {tensor::kKeypointLayer1, h8, w8, 64, 1, 1, 0, 64, maps[kHeadScratch], maps[kFusionInput], nullptr},
        {tensor::kKeypointLayer2, h8, w8, 64, 1, 1, 0, 64, maps[kFusionInput], maps[kHeadScratch], nullptr},
        {tensor::kKeypointLayer3, h8, w8, 64, 1, 1, 0, kLogitColumns, maps[kHeadScratch], nullptr,
         workspace.logits.get()},
    };
    for (const Convolution & c : heads) {
        const bool normalized = c.tensor == tensor::kKeypointLayer0;
        status = run_convolution(session, model, c, normalized ? workspace.keypoint_multipliers.get() : nullptr,
                                 normalized ? workspace.keypoint_biases.get() : nullptr);
        if (!status) return status;
    }
    timer.lap(stat::kHeadsMicroseconds);

    const int stride = (width + 31) / 32 * 32;
    const size_t heatmap_bytes = static_cast<size_t>(height) * stride * 4;
    const size_t ring_bytes = static_cast<size_t>(5) * stride * 4;
    if (heatmap_bytes + 4 * ring_bytes > session.vtcm_bytes()) return std::unexpected(runtime::Error::kNoMemory);
    auto * heatmap = reinterpret_cast<float *>(vtcm);
    CellHeatmapJob heatmap_job{workspace.logits.get(), heatmap, w8, stride};
    status = runtime::parallel_for(session, runtime::kHvxThreads, h8, softmax_cell_row, &heatmap_job);
    if (!status) return status;

    runtime::StageTimer reliability_timer(statistics);
    ReliabilityJob reliability_job{workspace.reliability_rows.get(), workspace.reliability.get(), cells};
    status = runtime::parallel_for(session, runtime::kHvxThreads,
                                   (cells + kReliabilityItemCells - 1) / kReliabilityItemCells, compute_reliability,
                                   &reliability_job);
    if (!status) return status;
    reliability_timer.lap(stat::kReliabilityMicroseconds);
    timer.lap(stat::kHeatmapMicroseconds);

    CandidateJob candidate_job{heatmap, height, width,     stride, arguments.threshold, workspace.reliability.get(),
                               {},      {},     &workspace};
    for (int thread = 0; thread < runtime::kHvxThreads; thread++) {
        candidate_job.rings[thread] = reinterpret_cast<float *>(vtcm + heatmap_bytes + thread * ring_bytes);
        candidate_job.row_buffers[thread] = workspace.row_buffers.get() + static_cast<size_t>(thread) * 2 * stride;
    }
    status = runtime::run_parallel(session, runtime::kHvxThreads, collect_candidates, &candidate_job);
    if (!status) return status;
    int candidates = 0;
    for (int thread = 0; thread < runtime::kHvxThreads; thread++) candidates += workspace.candidate_counts[thread];
    statistics[stat::kCandidates] = candidates;
    timer.lap(stat::kNmsMicroseconds);

    status = runtime::run_parallel(session, runtime::kHvxThreads, sort_band, &workspace);
    if (!status) return status;

    int band_next[runtime::kHvxThreads] = {};
    int count = 0;
    while (count < max_keypoints) {
        int best = -1;
        float best_score = 0.0f;
        for (int band = 0; band < runtime::kHvxThreads; band++) {
            if (band_next[band] == workspace.candidate_counts[band]) continue;
            const float score = workspace.sorted_scores[band][band_next[band]];
            if (best < 0 || score > best_score) {
                best = band;
                best_score = score;
            }
        }
        if (best < 0) break;

        const int32_t position = workspace.sorted_positions[best][band_next[best]++];
        features->keypoints[2 * count] = static_cast<float>(position & 0xffff);
        features->keypoints[2 * count + 1] = static_cast<float>(position >> 16);
        arguments.scores[count++] = best_score;
    }
    memcpy(arguments.keypoints, features->keypoints.get(), static_cast<size_t>(count) * 8);
    statistics[stat::kKeypoints] = count;
    timer.lap(stat::kKeypointSelectionMicroseconds);

    runtime::StageTimer normalization_timer(statistics);
    DescriptorJob descriptor_job{workspace.descriptor_map.get(), cells,     w8, h8, features->keypoints.get(), count,
                                 features->descriptors.get(),    &workspace};
    status = runtime::parallel_for(session, runtime::kHvxThreads, (cells + 31) / 32, normalize_cells, &descriptor_job);
    if (!status) return status;
    normalization_timer.lap(stat::kCellNormalizationMicroseconds);

    status =
        runtime::parallel_for(session, runtime::kHvxThreads, (count + kSampleItemKeypoints - 1) / kSampleItemKeypoints,
                              sample_descriptors, &descriptor_job);
    if (!status) return status;
    status = runtime::copy_memory_parallel(session, arguments.descriptors, features->descriptors.get(),
                                           static_cast<size_t>(count) * protocol::xfeat::kDescriptorDimensions * 2);
    if (!status) return status;
    features->count = count;
    timer.lap(stat::kDescriptorsMicroseconds);
    return {};
}

}  // namespace benzene::models::xfeat
