#include "dsp/models/loma/loma.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>

#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/prefetch.h"
#include "dsp/hvx/reductions.h"
#include "dsp/hvx/vector.h"
#include "dsp/kernels/convolution_int8/convolution_int8.h"
#include "dsp/kernels/depthwise_int8/depthwise_int8.h"
#include "dsp/models/shared/feature_slots.h"
#include "dsp/models/shared/layer_weights.h"
#include "dsp/models/shared/vision_transformer.h"
#include "dsp/runtime/cache.h"
#include "dsp/runtime/heap.h"
#include "dsp/runtime/stage_timer.h"
#include "dsp/runtime/status.h"
#include "dsp/runtime/synchronization.h"
#include "dsp/runtime/thread_pool.h"
#include "dsp/vector_math/functions.h"
#include "benzene/protocol.h"

namespace benzene::models::loma {
namespace {

namespace loma = protocol::loma;
namespace stat = protocol::loma::stat;
namespace refiner_part = protocol::loma::refiner_part;
namespace convolution_part = protocol::loma::convolution_part;
namespace depthwise_part = protocol::loma::depthwise_part;
namespace quantizer_part = protocol::loma::quantizer_part;
namespace tail_part = protocol::loma::dinov2_tail_part;
namespace convolution_int8 = kernels::convolution_int8;
using hvx::Mask;
using hvx::Vector;
using hvx::VectorPair;
using vision_transformer::kHidden;

// After every map: convolutions write whole output groups past a map's end, and vector reads run past it.
constexpr size_t kSlack = 64 * 1024;
constexpr int kDimensions = loma::kDescriptorDimensions;
constexpr int kMaximumKeypoints = 8192;

// One output index of device_model.bilinear_taps: the input indices i0, i1 and their weights l0, l1.
struct BilinearTap {
    int i0;
    int i1;
    float l0;
    float l1;
};

// The context upsampled into the next scale's input codes:
// - The rows, fp32 [height][width][row_stride], hold the out convolution's context part in columns [0, channels)
//   (channels a multiple of 32).
// - They are upsampled bilinearly (device_model.bilinear) to output_height x output_width and quantized,
//   q = trunc(clamp(f32(f32(v R) + Zq), 0, 255.5)), into channels [feature_channels, + channels) of the output rows
//   (output_channels codes per pixel).
// - With `feature`, its codes ([output_height][output_width][feature_channels], compact) are copied into channels
//   [0, feature_channels); without it, the caller writes them.
// - The channels after the context are zeroed.
struct ContextArguments {
    const float * rows = nullptr;
    int height = 0;
    int width = 0;
    int row_stride = 0;
    int channels = 0;
    const float * R = nullptr;   // [channels]
    const float * Zq = nullptr;  // [channels]
    const uint8_t * feature = nullptr;
    int feature_channels = 0;
    int output_height = 0;
    int output_width = 0;
    uint8_t * output = nullptr;
    int output_channels = 0;
};

// A refiner's working maps: [pixels][hidden] codes each, and x_and_x0 [pixels][2 hidden], which holds [x | x0] (the
// out layers' input) after run_refiner.
struct RefinerBuffers {
    uint8_t * middle;
    uint8_t * hidden_a;
    uint8_t * hidden_b;
    uint8_t * x0;
    uint8_t * x_and_x0;
};

struct RefinerMemory {
    runtime::HeapArray<uint8_t> middle;
    runtime::HeapArray<uint8_t> hidden_a;
    runtime::HeapArray<uint8_t> hidden_b;
    runtime::HeapArray<uint8_t> x0;
    runtime::HeapArray<uint8_t> x_and_x0;

    RefinerBuffers buffers() const { return {middle.get(), hidden_a.get(), hidden_b.get(), x0.get(), x_and_x0.get()}; }
};

struct RefinerArguments {
    const int32_t * references = nullptr;  // the refiner's first tensor (protocol.h loma::refiner_part order)
    const uint8_t * input = nullptr;       // [height][width][input_channels] codes
    int height = 0;
    int width = 0;
    int input_channels = 0;
    int hidden = 0;
    int hidden_blocks = 0;
    int context_columns = 0;  // out.context's N (padded to 32); 0: no context
    int delta_columns = 0;    // out.deltas' N; the logit: 32, written as rows of 8; 0: no dense deltas
    bool logit = false;
    float * context_rows = nullptr;  // [pixels][context_columns]
    float * delta_rows = nullptr;    // [pixels][delta_columns] (logit: [pixels][8])
    RefinerBuffers buffers{};
};

// Room for `bytes` and kSlack more.
template <class T>
bool allocate_with_slack(runtime::HeapArray<T> & buffer, size_t bytes) {
    return runtime::allocate(buffer, (bytes + kSlack + sizeof(T) - 1) / sizeof(T));
}

const int32_t * reference(const int32_t * references, int tensor) {
    return references + tensor * loma::kReferenceStride;
}

runtime::Result<LayerWeights> tensor(const runtime::Session & session, const int32_t * references, int index) {
    return layer_in_model(session, reference(references, index));
}

struct Quantizer {
    const float * R;
    const float * Zq;
};

runtime::Result<Quantizer> quantizer(const runtime::Session & session, const int32_t * references, int index) {
    const runtime::Result<LayerWeights> weights = tensor(session, references, index);
    if (!weights) return std::unexpected(weights.error());
    return Quantizer{weights->part<float>(quantizer_part::kReciprocalScales),
                     weights->part<float>(quantizer_part::kOffsets)};
}

constexpr int kMaximumSide = 4096;
constexpr int kMaximumContextChannels = 1024;

// One fp32 operation per statement (no fused multiply-add), rounded as in device_model.bilinear_taps.
void bilinear_taps(int inputs, int outputs, BilinearTap * taps) {
    const float scale = static_cast<float>(inputs) / static_cast<float>(outputs);

    for (int i = 0; i < outputs; i++) {
        float source = static_cast<float>(i) + 0.5f;
        source = source * scale;
        source = source - 0.5f;
        if (source < 0.0f) source = 0.0f;

        const float floor_source = floorf(source);
        const int i0 = static_cast<int>(floor_source);
        const float l1 = source - floor_source;
        const float l0 = 1.0f - l1;
        taps[i] = {i0, i0 + 1 < inputs ? i0 + 1 : inputs - 1, l0, l1};
    }
}

struct PoolJob {
    const uint8_t * input;
    int width;
    int channels;
    int input_stride;
    uint8_t * output;
};

inline Vector<uint8_t> window_maximum(const uint8_t * a, const uint8_t * b, const uint8_t * c, const uint8_t * d) {
    return max(max(hvx::load_unaligned(a), hvx::load_unaligned(b)),
               max(hvx::load_unaligned(c), hvx::load_unaligned(d)));
}

void pool_row(void * context, int y, int) {
    const auto * job = static_cast<const PoolJob *>(context);
    const int output_width = job->width / 2;
    const int channels = job->channels;
    const size_t stride = job->input_stride;
    const uint8_t * top = job->input + static_cast<size_t>(2 * y) * job->width * stride;
    const uint8_t * bottom = top + static_cast<size_t>(job->width) * stride;
    uint8_t * out = job->output + static_cast<size_t>(y) * output_width * channels;

    int x = 0;
    if (channels == 64) {
        const Mask first_half = Mask::first_bytes(64);
        for (; x + 1 < output_width; x += 2) {
            const uint8_t * a = top + static_cast<size_t>(2 * x) * stride;
            const uint8_t * c = bottom + static_cast<size_t>(2 * x) * stride;
            const Vector<uint8_t> left = window_maximum(a, a + stride, c, c + stride);
            const Vector<uint8_t> right =
                window_maximum(a + 2 * stride, a + 3 * stride, c + 2 * stride, c + 3 * stride);
            store_unaligned(out + static_cast<size_t>(x) * 64, select(first_half, left, rotate_right_bytes(right, 64)));
        }
    } else if (channels % 128 == 0) {
        for (; x < output_width; x++) {
            const uint8_t * a = top + static_cast<size_t>(2 * x) * stride;
            const uint8_t * c = bottom + static_cast<size_t>(2 * x) * stride;
            for (int k = 0; k < channels; k += 128) {
                store_unaligned(out + static_cast<size_t>(x) * channels + k,
                                window_maximum(a + k, a + stride + k, c + k, c + stride + k));
            }
        }
    }

    for (; x < output_width; x++) {
        const uint8_t * a = top + static_cast<size_t>(2 * x) * job->input_stride;
        const uint8_t * b = a + job->input_stride;
        const uint8_t * c = bottom + static_cast<size_t>(2 * x) * job->input_stride;
        const uint8_t * d = c + job->input_stride;
        uint8_t * o = out + static_cast<size_t>(x) * channels;
        for (int k = 0; k < channels; k += 64) {
            const Vector<uint8_t> m = window_maximum(a + k, b + k, c + k, d + k);
            alignas(128) uint8_t bytes[128];
            store(bytes, m);
            memcpy(o + k, bytes, 64);
        }
    }
}

constexpr int kConcatenateItemPixels = 256;

struct ConcatenationJob {
    const uint8_t * first;
    int first_channels;
    const uint8_t * second;
    int second_channels;
    int pixels;
    uint8_t * output;
    int output_channels;
};

void concatenate_pixels(void * context, int item, int) {
    const auto * job = static_cast<const ConcatenationJob *>(context);
    const int end =
        (item + 1) * kConcatenateItemPixels < job->pixels ? (item + 1) * kConcatenateItemPixels : job->pixels;
    const int tail = job->output_channels - job->first_channels - job->second_channels;

    int p = item * kConcatenateItemPixels;
    if (job->first_channels == 64 && job->second_channels == 64 && job->output_channels == 128) {
        const Mask first_half = Mask::first_bytes(64);
        for (; p < end; p++) {
            const Vector<uint8_t> first = hvx::load_unaligned(job->first + static_cast<size_t>(p) * 64);
            const Vector<uint8_t> second = hvx::load_unaligned(job->second + static_cast<size_t>(p) * 64);
            store_unaligned(job->output + static_cast<size_t>(p) * 128,
                            select(first_half, first, rotate_right_bytes(second, 64)));
        }
    } else if (job->first_channels == 32 && job->second_channels == 32 && job->output_channels == 64) {
        for (; p + 1 < end; p += 2) {
            const Vector<uint8_t> first = hvx::load_unaligned(job->first + static_cast<size_t>(p) * 32);
            const Vector<uint8_t> second = hvx::load_unaligned(job->second + static_cast<size_t>(p) * 32);
            store_unaligned(job->output + static_cast<size_t>(p) * 64, interleave_chunks<32>(second, first).low());
        }
    }

    for (; p < end; p++) {
        uint8_t * o = job->output + static_cast<size_t>(p) * job->output_channels;
        memcpy(o, job->first + static_cast<size_t>(p) * job->first_channels, job->first_channels);
        memcpy(o + job->first_channels, job->second + static_cast<size_t>(p) * job->second_channels,
               job->second_channels);
        if (tail > 0) memset(o + job->first_channels + job->second_channels, 0, tail);
    }
}

struct ContextJob {
    const ContextArguments * a;
    const BilinearTap * columns;
    const BilinearTap * rows;
    float * horizontal;
    size_t horizontal_floats;
    Vector<float> R[kMaximumContextChannels / 32];
    Vector<float> Zq[kMaximumContextChannels / 32];
};

void horizontal_context_row(const ContextJob & job, int r, float * out) {
    const ContextArguments & a = *job.a;
    const int groups = a.channels / 32;
    const float * row = a.rows + static_cast<size_t>(r) * a.width * a.row_stride;

    for (int j = 0; j < a.output_width; j++) {
        const BilinearTap column = job.columns[j];
        const Vector<float> a0 = Vector<float>::splat(column.l0);
        const Vector<float> a1 = Vector<float>::splat(column.l1);
        const float * x0 = row + static_cast<size_t>(column.i0) * a.row_stride;
        const float * x1 = row + static_cast<size_t>(column.i1) * a.row_stride;
        auto * h = hvx::vectors(out + static_cast<size_t>(j) * a.channels);
        for (int g = 0; g < groups; g++) {
            const Vector<float> v0 = hvx::load_unaligned(x0 + 32 * g);
            const Vector<float> v1 = hvx::load_unaligned(x1 + 32 * g);
            h[g] = a0 * v0 + a1 * v1;
        }
    }
}

inline Vector<int32_t> context_words(Vector<float> t0, Vector<float> t1, Vector<float> b0, Vector<float> b1,
                                     Vector<float> R, Vector<float> Zq) {
    const Vector<float> v = b0 * t0 + b1 * t1;
    Vector<float> q = v * R + Zq;
    q = min(max(q, Vector<float>::zero()), 255.5f);
    return truncate_to_int32(q);
}

void vertical_context_row(const ContextJob & job, int i, const float * h0, const float * h1) {
    const ContextArguments & a = *job.a;
    const BilinearTap row = job.rows[i];
    const Vector<float> b0 = Vector<float>::splat(row.l0);
    const Vector<float> b1 = Vector<float>::splat(row.l1);
    const Vector<int32_t> zero = Vector<int32_t>::zero();
    const int groups = a.channels / 32;
    const int tail = a.output_channels - a.feature_channels - a.channels;
    const auto * t0 = reinterpret_cast<const Vector<float> *>(h0);
    const auto * t1 = reinterpret_cast<const Vector<float> *>(h1);
    const size_t first_pixel = static_cast<size_t>(i) * a.output_width;

    if (a.feature != nullptr && a.feature_channels == 64 && a.channels == 32 && a.output_channels == 128) {
        const Mask first_half = Mask::first_bytes(64);
        for (int j = 0; j < a.output_width; j++) {
            const Vector<int32_t> words = context_words(t0[j], t1[j], b0, b1, job.R[0], job.Zq[0]);
            // The codes in bytes 0 .. 31, zeros after them.
            const Vector<uint8_t> codes =
                pack_to_uint8_saturating(reinterpret<int16_t>(zero), pack_to_int16_saturating(zero, words));
            const size_t pixel = first_pixel + j;
            const Vector<uint8_t> feature = hvx::load_unaligned(a.feature + pixel * 64);
            store_unaligned(a.output + pixel * 128, select(first_half, feature, rotate_right_bytes(codes, 64)));
        }
        return;
    }

    if (a.feature == nullptr && a.channels % 128 == 0 && a.feature_channels % 128 == 0 && tail == 0) {
        for (int j = 0; j < a.output_width; j++) {
            const size_t pixel = first_pixel + j;
            uint8_t * out = a.output + pixel * a.output_channels + a.feature_channels;
            const Vector<float> * u0 = t0 + static_cast<size_t>(j) * groups;
            const Vector<float> * u1 = t1 + static_cast<size_t>(j) * groups;
            for (int v = 0; v < groups / 4; v++) {
                Vector<int32_t> words[4];
#pragma clang loop unroll(full)
                for (int k = 0; k < 4; k++) {
                    const int g = 4 * v + k;
                    words[k] = context_words(u0[g], u1[g], b0, b1, job.R[g], job.Zq[g]);
                }
                store_unaligned(out + 128 * v, pack_to_uint8_saturating(pack_to_int16_saturating(words[3], words[2]),
                                                                        pack_to_int16_saturating(words[1], words[0])));
            }
        }
        return;
    }

    for (int j = 0; j < a.output_width; j++) {
        const size_t pixel = first_pixel + j;
        uint8_t * out = a.output + pixel * a.output_channels;
        if (a.feature != nullptr) memcpy(out, a.feature + pixel * a.feature_channels, a.feature_channels);
        alignas(128) uint8_t codes[kMaximumContextChannels];
        for (int g = 0; g < groups; g++) {
            const Vector<int32_t> words =
                context_words(t0[static_cast<size_t>(j) * groups + g], t1[static_cast<size_t>(j) * groups + g], b0, b1,
                              job.R[g], job.Zq[g]);
            // The codes in bytes 0 .. 31.
            const Vector<uint8_t> bytes =
                pack_to_uint8_saturating(reinterpret<int16_t>(zero), pack_to_int16_saturating(zero, words));
            store_unaligned(codes + 32 * g, bytes);
        }
        memcpy(out + a.feature_channels, codes, a.channels);
        if (tail > 0) memset(out + a.feature_channels + a.channels, 0, tail);
    }
}

void context_band(void * job_pointer, int thread) {
    const auto & job = *static_cast<const ContextJob *>(job_pointer);
    const ContextArguments & a = job.a[0];
    const int first = a.output_height * thread / runtime::kHvxThreads;
    const int end = a.output_height * (thread + 1) / runtime::kHvxThreads;
    float * slots[2] = {job.horizontal + static_cast<size_t>(thread) * 2 * job.horizontal_floats,
                        job.horizontal + (static_cast<size_t>(thread) * 2 + 1) * job.horizontal_floats};
    int held[2] = {-1, -1};
    const size_t input_row_bytes = static_cast<size_t>(a.width) * a.row_stride * 4;
    const size_t feature_row_bytes = static_cast<size_t>(a.output_width) * a.feature_channels;

    for (int i = first; i < end; i++) {
        const BilinearTap row = job.rows[i];
        if (i + 1 < end) {
            const BilinearTap next = job.rows[i + 1];
            hvx::prefetch_to_l2(a.rows + static_cast<size_t>(next.i1) * a.width * a.row_stride, 4096, 4096,
                                static_cast<uint32_t>((input_row_bytes + 4095) / 4096));
            if (a.feature != nullptr) {
                hvx::prefetch_to_l2(a.feature + static_cast<size_t>(i + 1) * feature_row_bytes, 4096, 4096,
                                    static_cast<uint32_t>((feature_row_bytes + 4095) / 4096));
            }
        }

        const float * h[2] = {nullptr, nullptr};
        const int needed[2] = {row.i0, row.i1};
        for (int n = 0; n < 2; n++) {
            const int s = held[0] == needed[n] ? 0 : held[1] == needed[n] ? 1 : -1;
            if (s >= 0) {
                h[n] = slots[s];
                continue;
            }

            const int victim = held[0] != needed[1 - n] ? 0 : 1;
            horizontal_context_row(job, needed[n], slots[victim]);
            held[victim] = needed[n];
            h[n] = slots[victim];
        }

        vertical_context_row(job, i, h[0], h[1]);
    }
}

// PyTorch's cubic convolution coefficients (A = -0.75) at t = 0.75 (even outputs: taps -2 .. 1) and t = 0.25 (odd
// outputs: taps -1 .. 2), exact in fp32.
constexpr float kEvenCoefficients[4] = {-0.03515625f, 0.26171875f, 0.87890625f, -0.10546875f};
constexpr float kOddCoefficients[4] = {-0.10546875f, 0.87890625f, 0.26171875f, -0.03515625f};

struct LogitJob {
    const float * previous;
    int width;
    int height;
    const float * deltas;
    int delta_stride;
    float * logits;
    float * horizontal;
};

void horizontal_row(void * context, int r, int) {
    const auto * job = static_cast<const LogitJob *>(context);
    const int input_width = job->width / 2;
    const float * row = job->previous + static_cast<size_t>(r) * input_width;

    alignas(128) float padded[kMaximumSide / 2 + 128];
    float * x = padded + 32;
    for (int k = -2; k < input_width + 34; k++) x[k] = row[k < 0 ? 0 : k < input_width ? k : input_width - 1];

    float * out = job->horizontal + static_cast<size_t>(r) * job->width;
    Vector<float> even_coefficients[4];
    Vector<float> odd_coefficients[4];
    for (int k = 0; k < 4; k++) {
        even_coefficients[k] = Vector<float>::splat(kEvenCoefficients[k]);
        odd_coefficients[k] = Vector<float>::splat(kOddCoefficients[k]);
    }

    for (int m = 0; m < input_width; m += 32) {
        const auto at = [&](int d) { return hvx::load_unaligned(x + m + d); };
        Vector<float> even = even_coefficients[0] * at(-2);
        Vector<float> odd = odd_coefficients[0] * at(-1);
        for (int k = 1; k < 4; k++) {
            even = even + even_coefficients[k] * at(k - 2);
            odd = odd + odd_coefficients[k] * at(k - 1);
        }

        const VectorPair<float> interleaved = interleave(odd, even);
        const int count = 2 * (input_width - m < 32 ? input_width - m : 32);
        if (count == 64) {
            store_unaligned(out + 2 * m, interleaved.low());
            store_unaligned(out + 2 * m + 32, interleaved.high());
        } else {
            alignas(128) float values[64];
            store(values, interleaved.low());
            store(values + 32, interleaved.high());
            memcpy(out + 2 * m, values, static_cast<size_t>(count) * 4);
        }
    }
}

void vertical_row(void * context, int i, int) {
    const auto * job = static_cast<const LogitJob *>(context);
    const int input_height = job->height / 2;
    const int m = i / 2;
    const bool even = i % 2 == 0;
    const float * coefficients = even ? kEvenCoefficients : kOddCoefficients;

    const float * rows[4];
    for (int k = 0; k < 4; k++) {
        int r = m + (even ? k - 2 : k - 1);
        r = r < 0 ? 0 : r >= input_height ? input_height - 1 : r;
        rows[k] = job->horizontal + static_cast<size_t>(r) * job->width;
    }
    Vector<float> c[4];
    for (int k = 0; k < 4; k++) c[k] = Vector<float>::splat(coefficients[k]);
    float * out = job->logits + static_cast<size_t>(i) * job->width;
    const float * delta_row = job->deltas + static_cast<size_t>(i) * job->width * job->delta_stride;

    for (int j = 0; j < job->width; j += 32) {
        const auto at = [&](int k) { return hvx::load_unaligned(rows[k] + j); };
        Vector<float> v = c[0] * at(0);
        for (int k = 1; k < 4; k++) v = v + c[k] * at(k);
        const int count = job->width - j < 32 ? job->width - j : 32;

        if (job->delta_stride == 8) {  // column 0 of 32 rows of 8: three rounds of deinterleave keep every eighth word
            const float * d = delta_row + static_cast<size_t>(j) * 8;
            Vector<float> quarters[4];
            for (int k = 0; k < 4; k++) {
                quarters[k] = deinterleave(hvx::load_unaligned(d + 64 * k + 32), hvx::load_unaligned(d + 64 * k)).low();
            }
            const Vector<float> halves[2] = {deinterleave(quarters[1], quarters[0]).low(),
                                             deinterleave(quarters[3], quarters[2]).low()};
            v = v + deinterleave(halves[1], halves[0]).low();
        } else {
            alignas(128) float deltas[32] = {};
            for (int k = 0; k < count; k++) deltas[k] = delta_row[static_cast<size_t>(j + k) * job->delta_stride];
            v = v + hvx::load(deltas);
        }

        if (count == 32) {
            store_unaligned(out + j, v);
        } else {
            alignas(128) float values[32];
            store(values, v);
            memcpy(out + j, values, static_cast<size_t>(count) * 4);
        }
    }
}

struct DescriptionJob {
    const float * previous;
    int previous_width;
    const float * deltas;
    int width;
    float * descriptions;
    const BilinearTap * columns;
    const BilinearTap * rows;
};

void description_row(void * context, int i, int) {
    const auto * job = static_cast<const DescriptionJob *>(context);
    const BilinearTap row = job->rows[i];
    const Vector<float> b0 = Vector<float>::splat(row.l0);
    const Vector<float> b1 = Vector<float>::splat(row.l1);

    for (int j = 0; j < job->width; j++) {
        const BilinearTap column = job->columns[j];
        const Vector<float> a0 = Vector<float>::splat(column.l0);
        const Vector<float> a1 = Vector<float>::splat(column.l1);
        const auto at = [&](int r, int c, int g) {
            return hvx::load(job->previous + (static_cast<size_t>(r) * job->previous_width + c) * 256 + 32 * g);
        };
        const size_t pixel = static_cast<size_t>(i) * job->width + j;
        const auto * d = hvx::vectors(job->deltas + pixel * 256);
        auto * out = hvx::vectors(job->descriptions + pixel * 256);

        for (int g = 0; g < 8; g++) {
            const Vector<float> t0 = a0 * at(row.i0, column.i0, g) + a1 * at(row.i0, column.i1, g);
            const Vector<float> t1 = a0 * at(row.i1, column.i0, g) + a1 * at(row.i1, column.i1, g);
            const Vector<float> up = b0 * t0 + b1 * t1;
            out[g] = up + d[g];
        }
    }
}

constexpr int kCopyLogitsItemPixels = 1024;

struct CopyLogitsJob {
    const float * deltas;
    int pixels;
    int delta_stride;
    float * logits;
};

void copy_logit_block(void * context, int item, int) {
    const auto * job = static_cast<const CopyLogitsJob *>(context);
    const int end = (item + 1) * kCopyLogitsItemPixels < job->pixels ? (item + 1) * kCopyLogitsItemPixels : job->pixels;
    for (int p = item * kCopyLogitsItemPixels; p < end; p++) {
        job->logits[p] = job->deltas[static_cast<size_t>(p) * job->delta_stride];
    }
}

runtime::Status max_pool(runtime::Session & session, const uint8_t * input, int height, int width, int channels,
                         int input_stride, uint8_t * output) {
    if (height < 2 || width < 2 || channels % 64 != 0 || input_stride < channels) {
        return std::unexpected(runtime::Error::kBadParameter);
    }
    PoolJob job{input, width, channels, input_stride, output};
    return runtime::parallel_for(session, runtime::kHvxThreads, height / 2, pool_row, &job);
}

runtime::Status concatenate(runtime::Session & session, const uint8_t * first, int first_channels,
                            const uint8_t * second, int second_channels, int pixels, uint8_t * output,
                            int output_channels) {
    if (first_channels + second_channels > output_channels) return std::unexpected(runtime::Error::kBadParameter);
    ConcatenationJob job{first, first_channels, second, second_channels, pixels, output, output_channels};
    const int items = (pixels + kConcatenateItemPixels - 1) / kConcatenateItemPixels;
    return runtime::parallel_for(session, runtime::kHvxThreads, items, concatenate_pixels, &job);
}

runtime::Status upsample_context(runtime::Session & session, const ContextArguments & arguments) {
    if (arguments.channels % 32 != 0 || arguments.channels > kMaximumContextChannels ||
        arguments.output_width > kMaximumSide || arguments.output_height > kMaximumSide ||
        arguments.feature_channels + arguments.channels > arguments.output_channels) {
        return std::unexpected(runtime::Error::kBadParameter);
    }
    const size_t horizontal_floats = static_cast<size_t>(arguments.output_width) * arguments.channels;
    if (static_cast<size_t>(runtime::kHvxThreads) * 2 * horizontal_floats * 4 > session.vtcm_bytes()) {
        return std::unexpected(runtime::Error::kNoMemory);
    }

    const runtime::HeapArray<BilinearTap> taps =
        runtime::allocate_array<BilinearTap>(arguments.output_width + arguments.output_height);
    const runtime::HeapObject<ContextJob> job = runtime::allocate_object<ContextJob>(128);
    if (!taps || !job) return std::unexpected(runtime::Error::kNoMemory);
    bilinear_taps(arguments.width, arguments.output_width, taps.get());
    bilinear_taps(arguments.height, arguments.output_height, taps.get() + arguments.output_width);

    job->a = &arguments;
    job->columns = taps.get();
    job->rows = taps.get() + arguments.output_width;
    job->horizontal = reinterpret_cast<float *>(session.vtcm());
    job->horizontal_floats = horizontal_floats;
    for (int g = 0; g < arguments.channels / 32; g++) {
        job->R[g] = hvx::load_unaligned(arguments.R + 32 * g);
        job->Zq[g] = hvx::load_unaligned(arguments.Zq + 32 * g);
    }
    return runtime::run_parallel(session, runtime::kHvxThreads, context_band, job.get());
}

runtime::Status accumulate_logits(runtime::Session & session, const float * previous, int height, int width,
                                  const float * deltas, int delta_stride, float * logits) {
    if (height % 2 != 0 || width % 2 != 0 || width > kMaximumSide || height > kMaximumSide) {
        return std::unexpected(runtime::Error::kBadParameter);
    }
    const runtime::HeapArray<float> horizontal =
        runtime::allocate_array<float>(static_cast<size_t>(height / 2) * width + 32);
    if (!horizontal) return std::unexpected(runtime::Error::kNoMemory);
    LogitJob job{previous, width, height, deltas, delta_stride, logits, horizontal.get()};
    const runtime::Status status =
        runtime::parallel_for(session, runtime::kHvxThreads, height / 2, horizontal_row, &job);
    if (!status) return status;
    return runtime::parallel_for(session, runtime::kHvxThreads, height, vertical_row, &job);
}

runtime::Status upsample_descriptions(runtime::Session & session, const float * previous, int previous_height,
                                      int previous_width, const float * deltas, int height, int width,
                                      float * descriptions) {
    const runtime::HeapArray<BilinearTap> taps = runtime::allocate_array<BilinearTap>(width + height);
    if (!taps) return std::unexpected(runtime::Error::kNoMemory);
    bilinear_taps(previous_width, width, taps.get());
    bilinear_taps(previous_height, height, taps.get() + width);
    DescriptionJob job{previous, previous_width, deltas, width, descriptions, taps.get(), taps.get() + width};
    return runtime::parallel_for(session, runtime::kHvxThreads, height, description_row, &job);
}

runtime::Status copy_logits(runtime::Session & session, const float * deltas, int pixels, int delta_stride,
                            float * logits) {
    CopyLogitsJob job{deltas, pixels, delta_stride, logits};
    const int items = (pixels + kCopyLogitsItemPixels - 1) / kCopyLogitsItemPixels;
    return runtime::parallel_for(session, runtime::kHvxThreads, items, copy_logit_block, &job);
}

constexpr int kMaximumColumns = 512;

runtime::Status run_convolution(runtime::Session & session, const convolution_int8::Arguments & arguments) {
    if (arguments.N <= kMaximumColumns) return kernels::convolution_int8::run(session, arguments);

    if (arguments.output_form != convolution_int8::OutputForm::kUint8Maps || arguments.pool || arguments.N % 128 != 0 ||
        arguments.uint8_outputs[1] != nullptr || arguments.N > arguments.uint8_output_channels) {
        return std::unexpected(runtime::Error::kBadParameter);
    }

    const int K = arguments.channels == 1 ? 32 : arguments.channels * arguments.taps;
    const size_t block_bytes = kernels::convolution_int8::weight_stream_bytes(K, 32);
    for (int first = 0; first < arguments.N; first += kMaximumColumns) {
        convolution_int8::Arguments chunk = arguments;
        chunk.N = arguments.N - first < kMaximumColumns ? arguments.N - first : kMaximumColumns;
        chunk.weights = arguments.weights + static_cast<size_t>(first / 32) * block_bytes;
        chunk.uint8_outputs[0] = arguments.uint8_outputs[0] + first;
        const runtime::Status status = kernels::convolution_int8::run(session, chunk);
        if (!status) return status;
    }
    return {};
}

runtime::Status pointwise(runtime::Session & session, const LayerWeights & weights, const uint8_t * input, int height,
                          int width, int channels, int columns, uint8_t * output, int output_channels,
                          float * rows = nullptr, int rows_columns = 0) {
    convolution_int8::Arguments a;
    a.input = input;
    a.height = height;
    a.width = width;
    a.channels = channels;
    a.taps = 1;
    a.weights = weights.part<uint8_t>(convolution_part::kWeights);
    a.N = columns;

    if (rows != nullptr) {
        a.output_form = convolution_int8::OutputForm::kRowsF32;
        a.rows_f32 = rows;
        a.rows_f32_columns = rows_columns;
    } else {
        a.uint8_outputs[0] = output;
        a.uint8_output_channels = output_channels;
    }
    return run_convolution(session, a);
}

runtime::Status depthwise(runtime::Session & session, const LayerWeights & weights, const uint8_t * input,
                          int input_stride, int height, int width, int channels, uint8_t * output) {
    kernels::depthwise_int8::Arguments a;
    a.input = input;
    a.height = height;
    a.width = width;
    a.channels = channels;
    a.weights = weights.part<int8_t>(depthwise_part::kWeights);
    a.initial = weights.part<int32_t>(depthwise_part::kInitial);
    a.output_multipliers = weights.part<float>(depthwise_part::kMultipliers);
    a.output_biases = weights.part<float>(depthwise_part::kBiases);
    a.zero_points = weights.part<uint8_t>(depthwise_part::kZeroPoints);
    a.output = output;
    a.input_stride = input_stride;
    return kernels::depthwise_int8::run(session, a);
}

// Adds the times of block1, the hidden blocks and the out layers to statistics[0], [1] and [2] (consecutive loma::stat
// entries), and that of the hidden blocks' depthwise layers (part of the hidden blocks) to *depthwise_statistic.
runtime::Status run_refiner(runtime::Session & session, const RefinerArguments & arguments, int32_t * statistics,
                            int32_t * depthwise_statistic) {
    const bool wide = arguments.hidden >= 128;
    const int both = 2 * arguments.hidden;
    runtime::Result<LayerWeights> weights;
    runtime::StageTimer timer(statistics);

    weights = tensor(session, arguments.references, refiner_part::kBlock1Conv1);
    if (!weights) return std::unexpected(weights.error());
    runtime::Status status =
        pointwise(session, *weights, arguments.input, arguments.height, arguments.width, arguments.input_channels,
                  arguments.hidden, arguments.buffers.middle, arguments.hidden);
    if (!status) return status;
    weights = tensor(session, arguments.references, refiner_part::kBlock1Conv2);
    if (!weights) return std::unexpected(weights.error());
    uint8_t * x0 = wide ? arguments.buffers.x_and_x0 + arguments.hidden : arguments.buffers.x0;
    status = pointwise(session, *weights, arguments.buffers.middle, arguments.height, arguments.width, arguments.hidden,
                       arguments.hidden, x0, wide ? both : arguments.hidden);
    if (!status) return status;
    timer.lap(0);

    const uint8_t * block_input = x0;
    int block_input_stride = wide ? both : arguments.hidden;
    for (int block = 0; block < arguments.hidden_blocks; block++) {
        weights = tensor(session, arguments.references, refiner_part::depthwise(block));
        if (!weights) return std::unexpected(weights.error());
        runtime::StageTimer depthwise_timer(depthwise_statistic);
        status = depthwise(session, *weights, block_input, block_input_stride, arguments.height, arguments.width,
                           arguments.hidden, arguments.buffers.middle);
        if (!status) return status;
        depthwise_timer.lap(0);

        const bool last = block + 1 == arguments.hidden_blocks;
        uint8_t * output = last && wide     ? arguments.buffers.x_and_x0
                           : block % 2 == 0 ? arguments.buffers.hidden_a
                                            : arguments.buffers.hidden_b;
        weights = tensor(session, arguments.references, refiner_part::pointwise(block));
        if (!weights) return std::unexpected(weights.error());
        status = pointwise(session, *weights, arguments.buffers.middle, arguments.height, arguments.width,
                           arguments.hidden, arguments.hidden, output, last && wide ? both : arguments.hidden);
        if (!status) return status;
        block_input = output;
        block_input_stride = arguments.hidden;
    }

    const int pixels = arguments.height * arguments.width;
    if (!wide) {
        status = concatenate(session, block_input, arguments.hidden, x0, arguments.hidden, pixels,
                             arguments.buffers.x_and_x0, both);
        if (!status) return status;
    }
    timer.lap(1);

    const bool with_context = arguments.context_columns > 0;
    if (with_context) {
        weights = tensor(session, arguments.references, refiner_part::context(arguments.hidden_blocks));
        if (!weights) return std::unexpected(weights.error());
        status = pointwise(session, *weights, arguments.buffers.x_and_x0, arguments.height, arguments.width, both,
                           arguments.context_columns, nullptr, 0, arguments.context_rows);
        if (!status) return status;
    }
    if (arguments.delta_columns > 0) {
        weights = tensor(session, arguments.references, refiner_part::deltas(arguments.hidden_blocks, with_context));
        if (!weights) return std::unexpected(weights.error());
        status = pointwise(session, *weights, arguments.buffers.x_and_x0, arguments.height, arguments.width, both,
                           arguments.delta_columns, nullptr, 0, arguments.delta_rows, arguments.logit ? 8 : 0);
        if (!status) return status;
    }
    timer.lap(2);
    return {};
}

struct DetectorWorkspace {
    size_t pixels;
    runtime::HeapArray<uint8_t> e0;
    // in2, in4, e2, e4 and e5 for the encoder and the coarser refiners, then in1 [pixels][128] over them.
    runtime::HeapArray<uint8_t> region;
    RefinerMemory refiner;
    runtime::HeapArray<float> context_rows;
    runtime::HeapArray<float> logit_rows;
    runtime::HeapArray<float> logits[3];
};

runtime::Status allocate_detector_workspace(DetectorWorkspace & workspace, size_t pixels) {
    if (workspace.pixels >= pixels) return {};

    workspace = DetectorWorkspace{};
    bool ok = allocate_with_slack(workspace.e0, pixels * 64);
    ok = ok && allocate_with_slack(workspace.region, pixels * 128 + 5 * kSlack);
    ok = ok && allocate_with_slack(workspace.refiner.middle, pixels * 32);
    ok = ok && allocate_with_slack(workspace.refiner.hidden_a, pixels * 32);
    ok = ok && allocate_with_slack(workspace.refiner.hidden_b, pixels * 32);
    ok = ok && allocate_with_slack(workspace.refiner.x0, pixels * 32);
    ok = ok && allocate_with_slack(workspace.refiner.x_and_x0, pixels * 64);
    ok = ok && allocate_with_slack(workspace.context_rows, pixels * 32);
    ok = ok && allocate_with_slack(workspace.logit_rows, pixels * 32);
    for (int s = 0; s < 3; s++)
        ok = ok && allocate_with_slack(workspace.logits[s], pixels * 4 / (s == 0 ? 64 : s == 1 ? 16 : 4));
    if (!ok) {
        workspace = DetectorWorkspace{};
        return std::unexpected(runtime::Error::kNoMemory);
    }
    workspace.pixels = pixels;
    return {};
}

runtime::Status detector_encoder_convolution(runtime::Session & session, const int32_t * references, int layer,
                                             const uint8_t * input, int height, int width, int channels, int zero_point,
                                             int columns, uint8_t * output, int output_channels) {
    const runtime::Result<LayerWeights> weights = tensor(session, references, loma::detector_encoder(layer));
    if (!weights) return std::unexpected(weights.error());

    convolution_int8::Arguments a;
    a.input = input;
    a.height = height;
    a.width = width;
    a.channels = channels;
    a.taps = 9;
    a.zero_point = zero_point;
    a.weights = weights->part<uint8_t>(convolution_part::kWeights);
    a.N = columns;
    a.uint8_outputs[0] = output;
    a.uint8_output_channels = output_channels;
    return run_convolution(session, a);
}

runtime::Status run_detector(runtime::Session & session, const int32_t * references, const uint8_t * gray, int height,
                             int width, int image_zero_point, float * logits, int32_t * statistics) {
    if (height % 8 != 0 || width % 8 != 0 || height < 16 || width < 16) {
        return std::unexpected(runtime::Error::kBadParameter);
    }
    DetectorWorkspace * workspace = session.state<DetectorWorkspace>();
    if (workspace == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const size_t pixels = static_cast<size_t>(height) * width;
    runtime::Status status = allocate_detector_workspace(*workspace, pixels);
    if (!status) return status;
    DetectorWorkspace & w = *workspace;

    runtime::StageTimer encoder_timer(statistics);
    const int h2 = height / 2, w2 = width / 2, h4 = height / 4, w4 = width / 4, h8 = height / 8, w8 = width / 8;
    uint8_t * in2 = w.region.get();
    uint8_t * in4 = in2 + pixels / 4 * 256 + kSlack;
    uint8_t * e2 = in4 + pixels / 16 * 512 + kSlack;
    uint8_t * e4 = e2 + pixels / 16 * 256 + kSlack;
    uint8_t * e5 = e4 + pixels / 64 * 512 + kSlack;
    uint8_t * pooled = w.refiner.middle.get();
    uint8_t * e0 = w.e0.get();

    status = detector_encoder_convolution(session, references, 0, gray, height, width, 1, image_zero_point, 64, e0, 64);
    if (!status) return status;
    status = max_pool(session, e0, height, width, 64, 64, pooled);
    if (!status) return status;
    status = detector_encoder_convolution(session, references, 1, pooled, h2, w2, 64, 0, 128, in2, 256);
    if (!status) return status;
    status = max_pool(session, in2, h2, w2, 128, 256, pooled);
    if (!status) return status;
    status = detector_encoder_convolution(session, references, 2, pooled, h4, w4, 128, 0, 256, e2, 256);
    if (!status) return status;
    status = detector_encoder_convolution(session, references, 3, e2, h4, w4, 256, 0, 256, in4, 512);
    if (!status) return status;
    status = max_pool(session, in4, h4, w4, 256, 512, pooled);
    if (!status) return status;
    status = detector_encoder_convolution(session, references, 4, pooled, h8, w8, 256, 0, 512, e4, 512);
    if (!status) return status;
    status = detector_encoder_convolution(session, references, 5, e4, h8, w8, 512, 0, 512, e5, 512);
    if (!status) return status;
    encoder_timer.lap(stat::kDetectorEncoderMicroseconds);

    struct Scale {
        int height, width, input_channels, hidden, context_columns;
        const uint8_t * input;
    };
    const Scale scales[loma::kDetectorScales] = {{h8, w8, 512, 512, 256, e5},
                                                 {h4, w4, 512, 256, 128, in4},
                                                 {h2, w2, 256, 64, 32, in2},
                                                 {height, width, 128, 32, 0, w.region.get()}};
    for (int s = 0; s < loma::kDetectorScales; s++) {
        const Scale & scale = scales[s];
        RefinerArguments r;
        r.references = reference(references, loma::detector_refiner(s));
        r.input = scale.input;
        r.height = scale.height;
        r.width = scale.width;
        r.input_channels = scale.input_channels;
        r.hidden = scale.hidden;
        r.hidden_blocks = loma::kDetectorHiddenBlocks;
        r.context_columns = scale.context_columns;
        r.delta_columns = 32;
        r.logit = true;
        r.context_rows = w.context_rows.get();
        r.delta_rows = w.logit_rows.get();
        r.buffers = w.refiner.buffers();
        status = run_refiner(session, r, statistics + stat::kDetectorBlock1Microseconds,
                             statistics + stat::kDetectorDepthwiseMicroseconds);
        if (!status) return status;

        runtime::StageTimer context_timer(statistics);
        float * scale_logits = s + 1 < loma::kDetectorScales ? w.logits[s].get() : logits;
        const int scale_pixels = scale.height * scale.width;
        status = s == 0 ? copy_logits(session, w.logit_rows.get(), scale_pixels, 8, scale_logits)
                        : accumulate_logits(session, w.logits[s - 1].get(), scale.height, scale.width,
                                            w.logit_rows.get(), 8, scale_logits);
        if (!status) return status;

        if (s + 1 < loma::kDetectorScales) {
            const runtime::Result<Quantizer> q =
                quantizer(session, references, loma::detector_context_quantizer(s + 1));
            if (!q) return std::unexpected(q.error());

            const Scale & next = scales[s + 1];
            ContextArguments c;
            c.rows = w.context_rows.get();
            c.height = scale.height;
            c.width = scale.width;
            c.row_stride = scale.context_columns;
            c.channels = scale.context_columns;
            c.R = q->R;
            c.Zq = q->Zq;
            c.feature = s + 2 == loma::kDetectorScales ? e0 : nullptr;
            c.feature_channels = s + 2 == loma::kDetectorScales ? 64 : next.input_channels / 2;
            c.output_height = next.height;
            c.output_width = next.width;
            c.output = const_cast<uint8_t *>(next.input);
            c.output_channels = next.input_channels;
            status = upsample_context(session, c);
            if (!status) return status;
        }
        context_timer.lap(stat::kDetectorContextMicroseconds);
    }
    return {};
}

constexpr float kLog2e = 1.4426950408889634f;
constexpr int kBuckets = 65536;
constexpr int kItemKeypoints = 32;

// e >= 0 orders like its bits: the histograms bucket the maxima by their upper 16 bits, and the sort keys
// bits << 32 | ~index in descending order run by value descending, then by raster index ascending.
struct KeypointWorkspace {
    size_t pixels;
    runtime::HeapArray<float> e;
    runtime::HeapArray<uint32_t> candidates;
    runtime::HeapArray<float> row_maxima;
    runtime::HeapArray<float> row_totals;
    runtime::HeapArray<uint32_t> histograms;
    uint32_t largest[runtime::kHvxThreads * 32];  // per thread, 128 bytes apart
    runtime::HeapArray<uint64_t> selection;
    runtime::HeapArray<uint64_t> row_lists;
    runtime::HeapArray<int> row_counts;
    runtime::HeapArray<int> chosen;
    runtime::HeapArray<float> values;
};

runtime::Status allocate_keypoint_workspace(KeypointWorkspace & workspace, size_t pixels, int height) {
    if (workspace.pixels >= pixels && workspace.row_totals) return {};

    workspace = KeypointWorkspace{};
    workspace.e = runtime::allocate_array<float>(pixels + 256);
    workspace.candidates = runtime::allocate_array<uint32_t>(pixels + 256);
    workspace.row_maxima = runtime::allocate_array<float>(kMaximumSide);
    workspace.row_totals = runtime::allocate_array<float>(kMaximumSide);
    workspace.histograms = runtime::allocate_array<uint32_t>(static_cast<size_t>(runtime::kHvxThreads) * kBuckets);
    workspace.selection = runtime::allocate_array<uint64_t>(pixels + 128);
    workspace.row_lists = runtime::allocate_array<uint64_t>(pixels + 128);
    workspace.row_counts = runtime::allocate_array<int>(kMaximumSide);
    workspace.chosen = runtime::allocate_array<int>(kMaximumKeypoints);
    workspace.values = runtime::allocate_array<float>(kMaximumKeypoints);
    if (!workspace.e || !workspace.candidates || !workspace.row_maxima || !workspace.row_totals ||
        !workspace.histograms || !workspace.selection || !workspace.row_lists || !workspace.row_counts ||
        !workspace.chosen || !workspace.values) {
        workspace = KeypointWorkspace{};
        return std::unexpected(runtime::Error::kNoMemory);
    }
    workspace.pixels = pixels;
    return {};
}

struct PassJob {
    const float * logits;
    int height;
    int width;
    float maximum;
    KeypointWorkspace * workspace;
};

inline Vector<float> load_lanes(const float * values, int count, Vector<float> fill) {
    const Vector<float> v = hvx::load_unaligned(values);
    return count < 32 ? select(Mask::first_bytes(4 * count), v, fill) : v;
}

void row_maximum(void * context, int y, int) {
    const auto * job = static_cast<const PassJob *>(context);
    KeypointWorkspace & workspace = *job->workspace;
    const float * row = job->logits + static_cast<size_t>(y) * job->width;
    const Vector<float> first = Vector<float>::splat(row[0]);

    Vector<float> m = first;
    for (int x = 0; x < job->width; x += 32) {
        m = max(m, load_lanes(row + x, job->width - x < 32 ? job->width - x : 32, first));
    }
    for (int shift = 16; shift >= 1; shift /= 2) m = max(m, rotate_right_bytes(m, 4 * shift));

    alignas(128) float lanes[32];
    store(lanes, m);
    workspace.row_maxima[y] = lanes[0];
}

// The row's total as device_model.lane_row_totals: 32 lane sums in order, then the rotate-and-add tree.
void exponential_row(void * context, int y, int) {
    const auto * job = static_cast<const PassJob *>(context);
    KeypointWorkspace & workspace = *job->workspace;
    const float * row = job->logits + static_cast<size_t>(y) * job->width;
    float * e = workspace.e.get() + static_cast<size_t>(y) * job->width;
    const Vector<float> maximum = Vector<float>::splat(job->maximum);
    const Vector<float> log2e = Vector<float>::splat(kLog2e);

    Vector<float> lanes = Vector<float>::zero();
    for (int x = 0; x < job->width; x += 32) {
        const int count = job->width - x < 32 ? job->width - x : 32;
        const Vector<float> d = load_lanes(row + x, count, Vector<float>::zero()) - maximum;
        Vector<float> v = vector_math::exp2(d * log2e);

        if (count < 32) {
            v = select(Mask::first_bytes(4 * count), v, Vector<float>::zero());
            alignas(128) float values[32];
            store(values, v);
            memcpy(e + x, values, static_cast<size_t>(count) * 4);
        } else {
            store_unaligned(e + x, v);
        }
        lanes = lanes + v;
    }
    for (int shift = 16; shift >= 1; shift /= 2) lanes = lanes + rotate_right_bytes(lanes, 4 * shift);

    alignas(128) float total[32];
    store(total, lanes);
    workspace.row_totals[y] = total[0];
}

// Pixels outside the map read 0, which no e is below: the 3x3 maximum is that of the pixels inside.
void nms_row(void * context, int y, int thread) {
    const auto * job = static_cast<const PassJob *>(context);
    KeypointWorkspace & workspace = *job->workspace;
    const int width = job->width;
    const float * rows[3];
    for (int k = 0; k < 3; k++) {
        const int r = y - 1 + k;
        rows[k] = r >= 0 && r < job->height ? workspace.e.get() + static_cast<size_t>(r) * width : nullptr;
    }
    uint32_t * histogram = workspace.histograms.get() + static_cast<size_t>(thread) * kBuckets;
    uint32_t * candidates = workspace.candidates.get() + static_cast<size_t>(y) * width;
    uint64_t * list = workspace.row_lists.get() + static_cast<size_t>(y) * width;
    int listed = 0;

    const Vector<float> zero = Vector<float>::zero();
    const Vector<uint32_t> one = Vector<uint32_t>::splat(1);
    Vector<float> previous[3] = {zero, zero, zero};
    Vector<float> current[3];
    for (int k = 0; k < 3; k++) {
        current[k] = rows[k] != nullptr ? load_lanes(rows[k], width < 32 ? width : 32, zero) : zero;
    }
    Vector<uint32_t> zero_maxima = reinterpret<uint32_t>(zero);
    for (int x = 0; x < width; x += 32) {
        const int count = width - x < 32 ? width - x : 32;
        Vector<float> m = current[1];
        Vector<float> next[3];
        for (int k = 0; k < 3; k++) {
            next[k] = rows[k] != nullptr && x + 32 < width
                          ? load_lanes(rows[k] + x + 32, width - x - 32 < 32 ? width - x - 32 : 32, zero)
                          : zero;
            const Vector<float> left = slide_up(current[k], previous[k], 4);
            const Vector<float> right = slide_down(next[k], current[k], 4);
            m = max(m, max(current[k], max(left, right)));
        }

        const Vector<float> centre = current[1];
        // e >= 0: equal values have equal bits.
        Mask maximum = reinterpret<int32_t>(centre) == reinterpret<int32_t>(m);
        if (count < 32) maximum = maximum & Mask::first_bytes(4 * count);
        const Vector<uint32_t> codes = reinterpret<uint32_t>(select(maximum, centre, zero));
        if (count == 32) {
            store_unaligned(candidates + x, codes);
        } else {
            alignas(128) uint32_t values[32];
            store(values, codes);
            memcpy(candidates + x, values, static_cast<size_t>(count) * 4);
        }

        const Mask at_zero = maximum & (reinterpret<int32_t>(centre) == reinterpret<int32_t>(zero));
        zero_maxima = zero_maxima + select(at_zero, one, reinterpret<uint32_t>(zero));
        const Mask positive = and_not(maximum, reinterpret<int32_t>(centre) == reinterpret<int32_t>(zero));
        if (any_lane(positive)) {
            alignas(128) uint32_t values[32];
            store(values, reinterpret<uint32_t>(select(positive, centre, zero)));
            for (int i = 0; i < count; i++) {
                if (values[i] != 0) {
                    histogram[values[i] >> 16]++;
                    if (values[i] > workspace.largest[32 * thread]) workspace.largest[32 * thread] = values[i];
                    const auto index = static_cast<uint32_t>(static_cast<size_t>(y) * width + x + i);
                    list[listed++] = (static_cast<uint64_t>(values[i]) << 32) | (0xFFFFFFFFu - index);
                }
            }
        }

        for (int k = 0; k < 3; k++) {
            previous[k] = current[k];
            current[k] = next[k];
        }
    }

    workspace.row_counts[y] = listed;
    alignas(128) uint32_t lane_counts[32];
    store(lane_counts, zero_maxima);
    uint32_t zeros = 0;
    for (int i = 0; i < 32; i++) zeros += lane_counts[i];
    histogram[0] += zeros;
}

struct PositionJob {
    const float * logits;
    int height;
    int width;
    int count;
    const int * chosen;
    const float * values;
    float total;
    float * positions;
    float * scores;
};

// The 3x3 logits read 0 outside the map, as in device_model.DeviceLoMa.keypoints (unfold's zero padding).
void position_block(void * context, int item, int) {
    const auto * job = static_cast<const PositionJob *>(context);
    const float * logits = job->logits;
    const int height = job->height;
    const int width = job->width;
    const int count = job->count;
    const int * chosen = job->chosen;
    const float * values = job->values;
    const float total = job->total;
    float * positions = job->positions;
    float * scores = job->scores;
    const float ux = 2.0f / static_cast<float>(width);
    const float uy = 2.0f / static_cast<float>(height);
    const int first = kItemKeypoints * item;

    alignas(128) float patch[9][32] = {};
    const int lanes = count - first < kItemKeypoints ? count - first : kItemKeypoints;
    for (int i = 0; i < lanes; i++) {
        const int p = chosen[first + i];
        const int y = p / width, x = p % width;
        for (int k = 0; k < 9; k++) {
            const int yy = y + k / 3 - 1, xx = x + k % 3 - 1;
            patch[k][i] =
                yy >= 0 && yy < height && xx >= 0 && xx < width ? logits[static_cast<size_t>(yy) * width + xx] : 0.0f;
        }
    }

    Vector<float> z[9];
    for (int k = 0; k < 9; k++) z[k] = hvx::load(patch[k]) * 2.0f;
    Vector<float> m = z[0];
    for (int k = 1; k < 9; k++) m = max(m, z[k]);
    Vector<float> w[9];
    for (int k = 0; k < 9; k++) w[k] = vector_math::exp2((z[k] - m) * kLog2e);
    Vector<float> total_w = w[0];
    for (int k = 1; k < 9; k++) total_w = total_w + w[k];

    Vector<float> num_x = Vector<float>::zero();
    Vector<float> num_y = Vector<float>::zero();
    for (int k = 0; k < 9; k++) {
        const float dx = ux * static_cast<float>(k % 3 - 1);
        const float dy = uy * static_cast<float>(k / 3 - 1);
        const Vector<float> tx = w[k] * dx;
        const Vector<float> ty = w[k] * dy;
        num_x = k == 0 ? tx : num_x + tx;
        num_y = k == 0 ? ty : num_y + ty;
    }

    alignas(128) float sums[3][32];
    store(sums[0], total_w);
    store(sums[1], num_x);
    store(sums[2], num_y);
    // Scalar for IEEE fp32 divisions.
    for (int i = 0; i < lanes; i++) {
        const int index = first + i;
        const int p = chosen[index];
        const int y = p / width, x = p % width;

        float xn = 2.0f * static_cast<float>(x);
        xn = xn + 1.0f;
        xn = xn / static_cast<float>(width);
        xn = xn - 1.0f;
        float yn = 2.0f * static_cast<float>(y);
        yn = yn + 1.0f;
        yn = yn / static_cast<float>(height);
        yn = yn - 1.0f;

        const float offset_x = sums[1][i] / sums[0][i];
        const float offset_y = sums[2][i] / sums[0][i];
        positions[2 * index] = xn + offset_x;
        positions[2 * index + 1] = yn + offset_y;
        scores[index] = values[index] / total;
    }
}

runtime::Status select_keypoints(runtime::Session & session, const float * logits, int height, int width, int count,
                                 float * positions, float * scores, int32_t * statistics) {
    if (width > kMaximumSide || height > kMaximumSide || count > kMaximumKeypoints || count < 1) {
        return std::unexpected(runtime::Error::kBadParameter);
    }
    const size_t P = static_cast<size_t>(height) * width;
    if (static_cast<size_t>(count) > P) return std::unexpected(runtime::Error::kBadParameter);
    KeypointWorkspace * kept = session.state<KeypointWorkspace>();
    if (kept == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    KeypointWorkspace & workspace = *kept;
    runtime::Status status = allocate_keypoint_workspace(workspace, P, height);
    if (!status) return status;

    runtime::StageTimer timer(statistics);
    PassJob job{logits, height, width, 0.0f, &workspace};
    status = runtime::parallel_for(session, runtime::kHvxThreads, height, row_maximum, &job);
    if (!status) return status;
    float maximum = workspace.row_maxima[0];
    for (int y = 1; y < height; y++) maximum = workspace.row_maxima[y] > maximum ? workspace.row_maxima[y] : maximum;
    job.maximum = maximum;

    status = runtime::parallel_for(session, runtime::kHvxThreads, height, exponential_row, &job);
    if (!status) return status;
    double total_double = 0.0;
    for (int y = 0; y < height; y++) total_double += static_cast<double>(workspace.row_totals[y]);
    const auto total = static_cast<float>(total_double);

    memset(workspace.histograms.get(), 0, static_cast<size_t>(runtime::kHvxThreads) * kBuckets * 4);
    memset(workspace.largest, 0, sizeof workspace.largest);
    status = runtime::parallel_for(session, runtime::kHvxThreads, height, nms_row, &job);
    if (!status) return status;
    timer.lap(stat::kSoftmaxMicroseconds);

    int64_t above = 0;
    uint32_t largest = 0;
    for (int t = 0; t < runtime::kHvxThreads; t++) {
        largest = workspace.largest[32 * t] > largest ? workspace.largest[32 * t] : largest;
    }
    int bucket = static_cast<int>(largest >> 16);
    for (; bucket >= 0; bucket--) {
        int64_t n = 0;
        for (int t = 0; t < runtime::kHvxThreads; t++) {
            n += workspace.histograms[static_cast<size_t>(t) * kBuckets + bucket];
        }
        if (above + n >= count) break;
        above += n;
    }

    size_t selected = 0;
    const uint32_t floor_bits = bucket >= 0 ? static_cast<uint32_t>(bucket) << 16 : 1u;
    if (floor_bits > 0) {
        for (int y = 0; y < height; y++) {
            const uint64_t * list = workspace.row_lists.get() + static_cast<size_t>(y) * width;
            for (int k = 0; k < workspace.row_counts[y]; k++) {
                if (static_cast<uint32_t>(list[k] >> 32) >= floor_bits) workspace.selection[selected++] = list[k];
            }
        }
    } else {
        for (size_t p = 0; p < P; p++) {
            workspace.selection[selected++] =
                (static_cast<uint64_t>(workspace.candidates[p]) << 32) | (0xFFFFFFFFu - p);
        }
    }
    std::sort(workspace.selection.get(), workspace.selection.get() + selected,
              [](uint64_t a, uint64_t b) { return a > b; });

    int * chosen = workspace.chosen.get();
    float * values = workspace.values.get();
    int n = 0;
    for (; n < count && static_cast<size_t>(n) < selected; n++) {
        const uint64_t key = workspace.selection[n];
        const auto bits = static_cast<uint32_t>(key >> 32);
        chosen[n] = static_cast<int32_t>(0xFFFFFFFFu - static_cast<uint32_t>(key));
        values[n] = std::bit_cast<float>(bits);
    }
    for (size_t p = 0; n < count && p < P; p++) {
        if (workspace.candidates[p] == 0) {
            chosen[n] = static_cast<int32_t>(p);
            values[n++] = 0.0f;
        }
    }
    timer.lap(stat::kSelectionMicroseconds);

    PositionJob position_job{logits, height, width, count, chosen, values, total, positions, scores};
    status = runtime::parallel_for(session, runtime::kHvxThreads, (count + kItemKeypoints - 1) / kItemKeypoints,
                                   position_block, &position_job);
    if (!status) return status;
    timer.lap(stat::kPositionMicroseconds);
    return {};
}

struct Dinov2Workspace {
    vision_transformer::Workspace blocks;
};

struct CodesJob {
    const vision_transformer::Workspace * workspace;
    const float * R;
    const float * Zq;
    uint8_t * codes;
};

// Token 0 is the class token, then the patches.
void quantize_token(void * context, int item, int) {
    const auto * job = static_cast<const CodesJob *>(context);
    const auto * x = hvx::vectors(job->workspace->x.get() + static_cast<size_t>(item + 1) * kHidden);
    const auto * R = hvx::vectors(job->R);
    const auto * Zq = hvx::vectors(job->Zq);
    const Vector<float> zero = Vector<float>::zero();
    const Vector<float> top = Vector<float>::splat(255.5f);
    auto * out = hvx::vectors(job->codes + static_cast<size_t>(item) * kHidden);

    for (int v = 0; v < kHidden / 128; v++) {
        Vector<int32_t> words[4];
        for (int k = 0; k < 4; k++) {
            const int i = 4 * v + k;
            Vector<float> q = x[i] * R[i] + Zq[i];
            q = min(max(q, zero), top);
            words[k] = truncate_to_int32(q);
        }
        out[v] = pack_to_uint8_saturating(pack_to_int16_saturating(words[3], words[2]),
                                          pack_to_int16_saturating(words[1], words[0]));
    }
}

runtime::Status run_dinov2(runtime::Session & session, const int32_t * references, const _Float16 * patches,
                           const float * positions, int rows, int columns, uint8_t * codes, int32_t * statistics) {
    runtime::StageTimer timer(statistics);
    const runtime::Result<LayerWeights> tail = tensor(session, references, loma::kDinov2Tail);
    const runtime::Result<LayerWeights> quantizer = tensor(session, references, loma::kDinov2OutputQuantizer);
    if (!tail || !quantizer) return std::unexpected(runtime::Error::kBadParameter);

    Dinov2Workspace * kept = session.state<Dinov2Workspace>();
    if (kept == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    vision_transformer::Workspace & workspace = kept->blocks;
    const int tokens = 1 + rows * columns;
    const int layout_rows = (tokens + 31) / 32 * 32;
    runtime::Status status = vision_transformer::allocate_workspace(&workspace, tokens, layout_rows);
    if (!status) return status;
    status = vision_transformer::set_geometry(&workspace, 1, tokens, rows, columns, 1);
    if (!status) return status;

    status = vision_transformer::embed(session, &workspace, tail->part<uint8_t>(tail_part::kPatchEmbedding), patches,
                                       loma::kDinov2PatchInputs, positions, tail->part<float>(tail_part::kClassToken));
    if (!status) return status;

    for (int block = 0; block < loma::kDinov2Blocks; block++) {
        const runtime::Result<LayerWeights> layer = tensor(session, references, loma::dinov2_block(block));
        if (!layer) return std::unexpected(layer.error());
        status = vision_transformer::run_block(session, *layer, protocol::vision_transformer::BlockKind::kDinov2,
                                               &workspace, statistics + stat::kDinov2BlockMicroseconds);
        if (!status) return status;
    }

    status = vision_transformer::normalize_patch_tokens(session, &workspace, tail->part<float>(tail_part::kNormGamma),
                                                        tail->part<float>(tail_part::kNormBeta));
    if (!status) return status;

    CodesJob job{&workspace, quantizer->part<float>(quantizer_part::kReciprocalScales),
                 quantizer->part<float>(quantizer_part::kOffsets), codes};
    status = runtime::parallel_for(session, runtime::kHvxThreads, rows * columns, quantize_token, &job);
    if (!status) return status;

    timer.lap(stat::kDinov2Microseconds);
    return {};
}

constexpr int kCorners = 4;
constexpr int kSources = 4;
constexpr int kGatherWidth = 64;

struct DescriptorWorkspace {
    size_t pixels;
    size_t patches;
    int keypoints;
    runtime::HeapArray<uint8_t> f1;
    // in2, t2, t4a and t4b for the encoder and the coarser refiners, then in1 [pixels][128] over them.
    runtime::HeapArray<uint8_t> fine;
    runtime::HeapArray<uint8_t> coarse;
    RefinerMemory refiner;
    runtime::HeapArray<float> context_rows;
    runtime::HeapArray<float> delta_rows;
    runtime::HeapArray<float> f14;
    runtime::HeapArray<float> f8;
    runtime::HeapArray<float> f4;
    runtime::HeapArray<uint8_t> dino_codes;
    runtime::HeapArray<uint8_t> gathered;
    runtime::HeapArray<float> sparse_rows;
    runtime::HeapArray<float> corner_rows;
};

size_t round_rows(size_t rows) {
    return (rows + kGatherWidth - 1) / kGatherWidth * kGatherWidth;
}

runtime::Status allocate_descriptor_workspace(DescriptorWorkspace & workspace, size_t pixels, size_t patches,
                                              int keypoints) {
    if (workspace.pixels >= pixels && workspace.patches >= patches && workspace.keypoints >= keypoints) return {};

    workspace = DescriptorWorkspace{};
    const size_t sparse = round_rows(static_cast<size_t>(keypoints) * kCorners * kSources);
    const size_t corners = round_rows(static_cast<size_t>(keypoints) * kCorners);
    const size_t hidden_map_bytes = pixels * 32 > patches * 768 ? pixels * 32 : patches * 768;
    bool ok = allocate_with_slack(workspace.f1, pixels * 64);
    ok = ok && allocate_with_slack(workspace.fine, pixels * 128 + 4 * kSlack);
    ok = ok && allocate_with_slack(workspace.coarse, pixels * 64 + 4 * kSlack);
    ok = ok && allocate_with_slack(workspace.refiner.middle, hidden_map_bytes);
    ok = ok && allocate_with_slack(workspace.refiner.hidden_a, hidden_map_bytes);
    ok = ok && allocate_with_slack(workspace.refiner.hidden_b, hidden_map_bytes);
    ok = ok && allocate_with_slack(workspace.refiner.x0, pixels * 32);
    ok = ok &&
         allocate_with_slack(workspace.refiner.x_and_x0, pixels * 64 > patches * 1536 ? pixels * 64 : patches * 1536);
    ok = ok && allocate_with_slack(workspace.context_rows, pixels * 32 > patches * 2048 ? pixels * 32 : patches * 2048);
    ok = ok && allocate_with_slack(workspace.delta_rows, pixels * 64 > patches * 1024 ? pixels * 64 : patches * 1024);
    ok = ok && allocate_with_slack(workspace.f14, patches * kDimensions * 4);
    ok = ok && allocate_with_slack(workspace.f8, pixels / 64 * kDimensions * 4 + kSlack);
    ok = ok && allocate_with_slack(workspace.f4, pixels / 16 * kDimensions * 4 + kSlack);
    ok = ok && allocate_with_slack(workspace.dino_codes, patches * 1024);
    ok = ok && allocate_with_slack(workspace.gathered, sparse * 128);
    ok = ok && allocate_with_slack(workspace.sparse_rows, sparse * kDimensions * 4);
    ok = ok && allocate_with_slack(workspace.corner_rows, corners * kDimensions * 4);
    if (!ok) {
        workspace = DescriptorWorkspace{};
        return std::unexpected(runtime::Error::kNoMemory);
    }
    workspace.pixels = pixels;
    workspace.patches = patches;
    workspace.keypoints = keypoints;
    return {};
}

runtime::Status descriptor_encoder_convolution(runtime::Session & session, const int32_t * references, int layer,
                                               const uint8_t * input, int height, int width, int channels, int columns,
                                               uint8_t * output, int output_channels) {
    const runtime::Result<LayerWeights> weights = tensor(session, references, loma::descriptor_encoder(layer));
    if (!weights) return std::unexpected(weights.error());

    convolution_int8::Arguments a;
    a.input = input;
    a.height = height;
    a.width = width;
    a.channels = channels;
    a.taps = 9;
    a.weights = weights->part<uint8_t>(convolution_part::kWeights);
    a.N = columns;
    a.uint8_outputs[0] = output;
    a.uint8_output_channels = output_channels;
    return run_convolution(session, a);
}

struct Sample {
    int corners[kCorners];
    float weights[kCorners];
};

struct SampleJob {
    const float * keypoints;
    int height;
    int width;
    int width2;
    const BilinearTap * rows21;
    const BilinearTap * columns21;
    Sample * samples;
    int * pixels2;
    int * pixels1;
};

// One fp32 operation per statement (no fused multiply-add), rounded as in device_model.DeviceLoMa.sample.
void sample_keypoint(void * context, int k, int) {
    const auto * job = static_cast<const SampleJob *>(context);
    const int width = job->width;
    const int height = job->height;
    const int width2 = job->width2;

    const float xn = job->keypoints[2 * k];
    const float yn = job->keypoints[2 * k + 1];
    float u = xn + 1.0f;
    u = u * static_cast<float>(width);
    u = u - 1.0f;
    const float ix = u / 2.0f;
    float v = yn + 1.0f;
    v = v * static_cast<float>(height);
    v = v - 1.0f;
    const float iy = v / 2.0f;

    const float x0 = floorf(ix);
    const float y0 = floorf(iy);
    const float wx1 = ix - x0;
    const float wy1 = iy - y0;
    const float wx0 = 1.0f - wx1;
    const float wy0 = 1.0f - wy1;
    const int xi = static_cast<int>(x0);
    const int yi = static_cast<int>(y0);

    Sample & s = job->samples[k];
    const int dys[kCorners] = {0, 0, 1, 1};
    const int dxs[kCorners] = {0, 1, 0, 1};
    const float wys[kCorners] = {wy0, wy0, wy1, wy1};
    const float wxs[kCorners] = {wx0, wx1, wx0, wx1};
    for (int c = 0; c < kCorners; c++) {
        const int y = yi + dys[c];
        const int x = xi + dxs[c];
        const bool inside = y >= 0 && y < height && x >= 0 && x < width;
        s.corners[c] = inside ? y * width + x : -1;
        s.weights[c] = wys[c] * wxs[c];
        job->pixels1[k * kCorners + c] = inside ? y * width + x : 0;

        const BilinearTap row = job->rows21[inside ? y : 0];
        const BilinearTap column = job->columns21[inside ? x : 0];
        int * sources = job->pixels2 + (k * kCorners + c) * kSources;
        sources[0] = row.i0 * width2 + column.i0;
        sources[1] = row.i0 * width2 + column.i1;
        sources[2] = row.i1 * width2 + column.i0;
        sources[3] = row.i1 * width2 + column.i1;
    }
}

struct GatherJob {
    const uint8_t * map;
    int channels;
    const int * pixels;
    int count;
    uint8_t * rows;
};

void gather_rows(void * context, int item, int) {
    const auto * job = static_cast<const GatherJob *>(context);
    const int first = item * kGatherWidth;

    for (int r = first; r < first + kGatherWidth; r++) {
        uint8_t * out = job->rows + static_cast<size_t>(r) * job->channels;
        if (r < job->count) {
            memcpy(out, job->map + static_cast<size_t>(job->pixels[r]) * job->channels, job->channels);
        } else {
            memset(out, 0, job->channels);
        }
    }
}

runtime::Status sparse_deltas(runtime::Session & session, const DescriptorWorkspace & workspace,
                              const int32_t * references, int refiner, bool with_context, const uint8_t * map,
                              int channels, const int * pixels, int count, float * rows) {
    const runtime::Result<LayerWeights> weights =
        tensor(session, references,
               loma::descriptor_refiner(refiner) + refiner_part::deltas(loma::kDescriptorHiddenBlocks, with_context));
    if (!weights) return std::unexpected(weights.error());

    GatherJob gather{map, channels, pixels, count, workspace.gathered.get()};
    const int items = (count + kGatherWidth - 1) / kGatherWidth;
    const runtime::Status gathered = runtime::parallel_for(session, runtime::kHvxThreads, items, gather_rows, &gather);
    if (!gathered) return gathered;

    convolution_int8::Arguments a;
    a.input = workspace.gathered.get();
    a.height = items;
    a.width = kGatherWidth;
    a.channels = channels;
    a.taps = 1;
    a.weights = weights->part<uint8_t>(convolution_part::kWeights);
    a.N = kDimensions;
    a.output_form = convolution_int8::OutputForm::kRowsF32;
    a.rows_f32 = rows;
    return run_convolution(session, a);
}

// Per keypoint, F1 = upsampled F2 + D1 at its corners, from F2 = upsampled F4 + D2 at their 1/2 pixels; the corners
// weighted and summed in order (device_model.DeviceLoMa.sample).
struct CombineJob {
    const Sample * samples;
    int count;
    int width;
    int width4;
    const BilinearTap * rows21;
    const BilinearTap * columns21;
    const BilinearTap * rows42;
    const BilinearTap * columns42;
    const float * f4;
    const float * d2;
    const float * d1;
    float * descriptors;
};

enum class Region { kHalfDeltas, kQuarterPixels, kCornerDeltas };

void prefetch_keypoint(const CombineJob * job, int k, Region region) {
    if (k >= job->count) return;

    if (region == Region::kHalfDeltas) {
        hvx::prefetch_to_l2(job->d2 + static_cast<size_t>(k) * kCorners * kSources * kDimensions, 4096, 4096,
                            kCorners * kSources * kDimensions * 4 / 4096);
    } else if (region == Region::kCornerDeltas) {
        hvx::prefetch_to_l2(job->d1 + static_cast<size_t>(k) * kCorners * kDimensions, 4096, 4096,
                            kCorners * kDimensions * 4 / 4096);
    } else {
        const Sample & s = job->samples[k];
        for (int c = 0; c < kCorners; c++) {
            if (s.corners[c] < 0) continue;

            const BilinearTap row = job->rows21[s.corners[c] / job->width];
            const BilinearTap column = job->columns21[s.corners[c] % job->width];
            const int first_row = job->rows42[row.i0].i0;
            const int last_row = job->rows42[row.i1].i1;
            const int first_column = job->columns42[column.i0].i0;
            const int last_column = job->columns42[column.i1].i1;
            hvx::prefetch_to_l2(job->f4 + (static_cast<size_t>(first_row) * job->width4 + first_column) * kDimensions,
                                static_cast<uint32_t>(last_column - first_column + 1) * kDimensions * 4,
                                static_cast<uint32_t>(job->width4) * kDimensions * 4,
                                static_cast<uint32_t>(last_row - first_row + 1));
            break;
        }
    }
}

void combine_keypoint(void * context, int k, int) {
    const auto * job = static_cast<const CombineJob *>(context);
    const int ahead = k + 2 * runtime::kHvxThreads;
    prefetch_keypoint(job, ahead, Region::kHalfDeltas);

    const Sample & s = job->samples[k];
    Vector<float> total[8];
    bool started = false;
    for (int c = 0; c < kCorners; c++) {
        if (s.corners[c] < 0) continue;

        const int y = s.corners[c] / job->width;
        const int x = s.corners[c] % job->width;
        const BilinearTap row = job->rows21[y];
        const BilinearTap column = job->columns21[x];

        Vector<float> f2[kSources][8];
        const int rs[kSources] = {row.i0, row.i0, row.i1, row.i1};
        const int cs[kSources] = {column.i0, column.i1, column.i0, column.i1};
        for (int q = 0; q < kSources; q++) {
            const BilinearTap row4 = job->rows42[rs[q]];
            const BilinearTap column4 = job->columns42[cs[q]];
            const Vector<float> a0 = Vector<float>::splat(column4.l0);
            const Vector<float> a1 = Vector<float>::splat(column4.l1);
            const Vector<float> b0 = Vector<float>::splat(row4.l0);
            const Vector<float> b1 = Vector<float>::splat(row4.l1);
            const auto * d2 =
                hvx::vectors(job->d2 + ((static_cast<size_t>(k) * kCorners + c) * kSources + q) * kDimensions);
            const auto at = [&](int r, int cc, int g) {
                return hvx::load(job->f4 + (static_cast<size_t>(r) * job->width4 + cc) * kDimensions + 32 * g);
            };

            for (int g = 0; g < 8; g++) {
                const Vector<float> t0 = a0 * at(row4.i0, column4.i0, g) + a1 * at(row4.i0, column4.i1, g);
                const Vector<float> t1 = a0 * at(row4.i1, column4.i0, g) + a1 * at(row4.i1, column4.i1, g);
                const Vector<float> up = b0 * t0 + b1 * t1;
                f2[q][g] = up + d2[g];
            }
        }

        const Vector<float> a0 = Vector<float>::splat(column.l0);
        const Vector<float> a1 = Vector<float>::splat(column.l1);
        const Vector<float> b0 = Vector<float>::splat(row.l0);
        const Vector<float> b1 = Vector<float>::splat(row.l1);
        const Vector<float> weight = Vector<float>::splat(s.weights[c]);
        const auto * d1 = hvx::vectors(job->d1 + (static_cast<size_t>(k) * kCorners + c) * kDimensions);
        for (int g = 0; g < 8; g++) {
            const Vector<float> t0 = a0 * f2[0][g] + a1 * f2[1][g];
            const Vector<float> t1 = a0 * f2[2][g] + a1 * f2[3][g];
            const Vector<float> up = b0 * t0 + b1 * t1;
            const Vector<float> f1 = up + d1[g];
            const Vector<float> term = weight * f1;
            total[g] = started ? total[g] + term : term;
        }
        if (c == 1) prefetch_keypoint(job, ahead, Region::kQuarterPixels);
        started = true;
    }

    prefetch_keypoint(job, ahead, Region::kCornerDeltas);
    float * out = job->descriptors + static_cast<size_t>(k) * kDimensions;
    for (int g = 0; g < 8; g++) store_unaligned(out + 32 * g, started ? total[g] : Vector<float>::zero());
}

runtime::Status run_descriptor(runtime::Session & session, const int32_t * references, const uint8_t * gray, int height,
                               int width, const _Float16 * patches, const float * positions, const float * keypoints,
                               int count, float * descriptors, int32_t * statistics) {
    if (height % 14 != 0 || width % 14 != 0 || height < 28 || width < 28 || count < 1) {
        return std::unexpected(runtime::Error::kBadParameter);
    }
    const int rows14 = height / 14, columns14 = width / 14;
    const int h2 = height / 2, w2 = width / 2, h4 = h2 / 2, w4 = w2 / 2, h8 = h4 / 2, w8 = w4 / 2;
    const size_t P = static_cast<size_t>(height) * width;
    DescriptorWorkspace * workspace = session.state<DescriptorWorkspace>();
    if (workspace == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    runtime::Status status =
        allocate_descriptor_workspace(*workspace, P, static_cast<size_t>(rows14) * columns14, count);
    if (!status) return status;
    DescriptorWorkspace & w = *workspace;

    runtime::StageTimer encoder_timer(statistics);
    const size_t P2 = static_cast<size_t>(h2) * w2, P4 = static_cast<size_t>(h4) * w4,
                 P8 = static_cast<size_t>(h8) * w8;
    uint8_t * in2 = w.fine.get();
    uint8_t * t2 = in2 + P2 * 256 + kSlack;
    uint8_t * t4a = t2 + P2 * 128 + kSlack;
    uint8_t * t4b = t4a + P4 * 256 + kSlack;
    uint8_t * in4 = w.coarse.get();
    uint8_t * in8 = in4 + P4 * 512 + kSlack;
    uint8_t * t8a = in8 + P8 * 1024 + kSlack;
    uint8_t * t8b = t8a + P8 * 512 + kSlack;
    uint8_t * e0 = w.refiner.x_and_x0.get();
    uint8_t * pooled = w.refiner.middle.get();
    uint8_t * f1 = w.f1.get();

    status = descriptor_encoder_convolution(session, references, 0, gray, height, width, 1, 64, e0, 64);
    if (!status) return status;
    status = descriptor_encoder_convolution(session, references, 1, e0, height, width, 64, 64, f1, 64);
    if (!status) return status;
    status = max_pool(session, f1, height, width, 64, 64, pooled);
    if (!status) return status;
    status = descriptor_encoder_convolution(session, references, 2, pooled, h2, w2, 64, 128, t2, 128);
    if (!status) return status;
    status = descriptor_encoder_convolution(session, references, 3, t2, h2, w2, 128, 128, in2, 256);
    if (!status) return status;
    status = max_pool(session, in2, h2, w2, 128, 256, pooled);
    if (!status) return status;
    status = descriptor_encoder_convolution(session, references, 4, pooled, h4, w4, 128, 256, t4a, 256);
    if (!status) return status;
    status = descriptor_encoder_convolution(session, references, 5, t4a, h4, w4, 256, 256, t4b, 256);
    if (!status) return status;
    status = descriptor_encoder_convolution(session, references, 6, t4b, h4, w4, 256, 256, t4a, 256);
    if (!status) return status;
    status = descriptor_encoder_convolution(session, references, 7, t4a, h4, w4, 256, 256, in4, 512);
    if (!status) return status;
    status = max_pool(session, in4, h4, w4, 256, 512, pooled);
    if (!status) return status;
    status = descriptor_encoder_convolution(session, references, 8, pooled, h8, w8, 256, 512, t8a, 512);
    if (!status) return status;
    status = descriptor_encoder_convolution(session, references, 9, t8a, h8, w8, 512, 512, t8b, 512);
    if (!status) return status;
    status = descriptor_encoder_convolution(session, references, 10, t8b, h8, w8, 512, 512, t8a, 512);
    if (!status) return status;
    status = descriptor_encoder_convolution(session, references, 11, t8a, h8, w8, 512, 512, in8, 1024);
    if (!status) return status;
    encoder_timer.lap(stat::kDescriptorEncoderMicroseconds);

    status = run_dinov2(session, references, patches, positions, rows14, columns14, w.dino_codes.get(), statistics);
    if (!status) return status;

    const runtime::HeapArray<BilinearTap> taps = runtime::allocate_array<BilinearTap>(height + width + h2 + w2);
    const runtime::HeapArray<Sample> samples = runtime::allocate_array<Sample>(count);
    const runtime::HeapArray<int> pixels2 =
        runtime::allocate_array<int>(static_cast<size_t>(count) * kCorners * kSources);
    const runtime::HeapArray<int> pixels1 = runtime::allocate_array<int>(static_cast<size_t>(count) * kCorners);
    if (!taps || !samples || !pixels2 || !pixels1) return std::unexpected(runtime::Error::kNoMemory);
    BilinearTap * rows21 = taps.get();
    BilinearTap * columns21 = rows21 + height;
    BilinearTap * rows42 = columns21 + width;
    BilinearTap * columns42 = rows42 + h2;
    bilinear_taps(h2, height, rows21);
    bilinear_taps(w2, width, columns21);
    bilinear_taps(h4, h2, rows42);
    bilinear_taps(w4, w2, columns42);
    SampleJob sample_job{keypoints, height, width, w2, rows21, columns21, samples.get(), pixels2.get(), pixels1.get()};
    status = runtime::parallel_for(session, runtime::kHvxThreads, count, sample_keypoint, &sample_job);
    if (!status) return status;

    struct Scale {
        int height, width, input_channels, hidden, context_columns, delta_columns;
        const uint8_t * input;
    };
    const Scale scales[loma::kDescriptorScales] = {{rows14, columns14, 1024, 768, 512, 256, w.dino_codes.get()},
                                                   {h8, w8, 1024, 512, 256, 256, in8},
                                                   {h4, w4, 512, 256, 128, 256, in4},
                                                   {h2, w2, 256, 64, 32, 0, in2},
                                                   {height, width, 128, 32, 0, 0, w.fine.get()}};
    float * maps[3] = {w.f14.get(), w.f8.get(), w.f4.get()};
    for (int s = 0; s < loma::kDescriptorScales; s++) {
        const Scale & scale = scales[s];
        RefinerArguments r;
        r.references = reference(references, loma::descriptor_refiner(s));
        r.input = scale.input;
        r.height = scale.height;
        r.width = scale.width;
        r.input_channels = scale.input_channels;
        r.hidden = scale.hidden;
        r.hidden_blocks = loma::kDescriptorHiddenBlocks;
        r.context_columns = scale.context_columns;
        r.delta_columns = scale.delta_columns;
        r.context_rows = w.context_rows.get();
        r.delta_rows = w.delta_rows.get();
        r.buffers = w.refiner.buffers();
        status = run_refiner(session, r, statistics + stat::kDescriptorBlock1Microseconds,
                             statistics + stat::kDescriptorDepthwiseMicroseconds);
        if (!status) return status;

        runtime::StageTimer scale_timer(statistics);
        const int scale_pixels = scale.height * scale.width;
        if (s == 0) {
            memcpy(w.f14.get(), w.delta_rows.get(), static_cast<size_t>(scale_pixels) * kDimensions * 4);
        } else if (s < 3) {
            const Scale & previous = scales[s - 1];
            status = upsample_descriptions(session, maps[s - 1], previous.height, previous.width, w.delta_rows.get(),
                                           scale.height, scale.width, maps[s]);
            if (!status) return status;
        }

        if (s + 1 < loma::kDescriptorScales) {
            const runtime::Result<Quantizer> q =
                quantizer(session, references, loma::descriptor_context_quantizer(s + 1));
            if (!q) return std::unexpected(q.error());

            const Scale & next = scales[s + 1];
            ContextArguments c;
            c.rows = w.context_rows.get();
            c.height = scale.height;
            c.width = scale.width;
            c.row_stride = scale.context_columns;
            c.channels = scale.context_columns;
            c.R = q->R;
            c.Zq = q->Zq;
            c.feature = s + 2 == loma::kDescriptorScales ? f1 : nullptr;
            c.feature_channels = s + 2 == loma::kDescriptorScales ? 64 : next.input_channels / 2;
            c.output_height = next.height;
            c.output_width = next.width;
            c.output = const_cast<uint8_t *>(next.input);
            c.output_channels = next.input_channels;
            status = upsample_context(session, c);
            if (!status) return status;
        }
        scale_timer.lap(stat::kDescriptorContextMicroseconds);

        if (s == 3) {
            status = sparse_deltas(session, w, references, 3, true, w.refiner.x_and_x0.get(), 128, pixels2.get(),
                                   count * kCorners * kSources, w.sparse_rows.get());
            if (!status) return status;
        }
        if (s == 4) {
            status = sparse_deltas(session, w, references, 4, false, w.refiner.x_and_x0.get(), 64, pixels1.get(),
                                   count * kCorners, w.corner_rows.get());
            if (!status) return status;
        }
        if (s >= 3) scale_timer.lap(stat::kSparseMicroseconds);
    }

    runtime::StageTimer combine_timer(statistics);
    CombineJob combine_job{samples.get(),       count,      width,     w4,         rows21,
                           columns21,           rows42,     columns42, w.f4.get(), w.sparse_rows.get(),
                           w.corner_rows.get(), descriptors};
    prefetch_keypoint(&combine_job, 0, Region::kHalfDeltas);
    status = runtime::parallel_for(session, runtime::kHvxThreads, count, combine_keypoint, &combine_job);
    combine_timer.lap(stat::kSparseMicroseconds);
    return status;
}

constexpr int kItemDescriptions = 32;

struct Workspace {
    runtime::GrowingArray<float> logits;
};

struct HalfJob {
    const float * descriptions;
    _Float16 * halves;
    int count;
};

void convert_descriptions(void * context, int item, int) {
    const auto * job = static_cast<const HalfJob *>(context);
    const int first = item * kItemDescriptions;

    for (int k = first; k < first + kItemDescriptions && k < job->count; k++) {
        const float * in = job->descriptions + static_cast<size_t>(k) * kDimensions;
        _Float16 * out = job->halves + static_cast<size_t>(k) * kDimensions;
        for (int v = 0; v < kDimensions / 64; v++) {
            store_unaligned(out + 64 * v,
                            to_float16(hvx::load_unaligned(in + 64 * v), hvx::load_unaligned(in + 64 * v + 32)));
        }
    }
}

// One call at a time: the calls share the session's state.
struct CallGuard {
    std::atomic<int> in_use;
};

}  // namespace

runtime::Status run(runtime::Session & session, const Arguments & arguments) {
    runtime::StageTimer timer(arguments.statistics);
    if (arguments.count < 1 || arguments.count > kMaximumKeypoints || arguments.detector_zero_point < 0 ||
        arguments.detector_zero_point > 255) {
        return std::unexpected(runtime::Error::kBadParameter);
    }
    Features * features = feature_slot(session, arguments.slot);
    if (features == nullptr) return std::unexpected(runtime::Error::kBadParameter);
    CallGuard * guard = session.state<CallGuard>();
    if (guard == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const runtime::ExclusiveUse use(guard->in_use);
    if (!use.taken()) return std::unexpected(runtime::Error::kBadParameter);
    Workspace * workspace = session.state<Workspace>();
    if (workspace == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const size_t pixels = static_cast<size_t>(arguments.detector_height) * arguments.detector_width;
    float * logits = workspace->logits.reserve(pixels + 1024);
    if (logits == nullptr) return std::unexpected(runtime::Error::kNoMemory);

    runtime::invalidate_host_input(arguments.detector_image, pixels);
    runtime::Status status =
        run_detector(session, arguments.references, arguments.detector_image, arguments.detector_height,
                     arguments.detector_width, arguments.detector_zero_point, logits, arguments.statistics);
    if (!status) return status;

    status = select_keypoints(session, logits, arguments.detector_height, arguments.detector_width, arguments.count,
                              arguments.keypoints, arguments.scores, arguments.statistics);
    if (!status) return status;

    status = reserve_features(features, arguments.count, kDimensions);
    if (!status) return status;
    features->count = 0;

    const size_t descriptor_pixels = static_cast<size_t>(arguments.descriptor_height) * arguments.descriptor_width;
    runtime::invalidate_host_input(arguments.descriptor_image, descriptor_pixels);
    status = run_descriptor(session, arguments.references, arguments.descriptor_image, arguments.descriptor_height,
                            arguments.descriptor_width, arguments.patches, arguments.positions, arguments.keypoints,
                            arguments.count, arguments.descriptors, arguments.statistics);
    if (!status) return status;

    HalfJob half_job{arguments.descriptors, features->descriptors.get(), arguments.count};
    status = runtime::parallel_for(session, runtime::kHvxThreads,
                                   (arguments.count + kItemDescriptions - 1) / kItemDescriptions, convert_descriptions,
                                   &half_job);
    if (!status) return status;

    features->count = arguments.count;
    features->height = arguments.detector_height;
    features->width = arguments.detector_width;
    memcpy(features->keypoints.get(), arguments.keypoints, static_cast<size_t>(arguments.count) * 2 * 4);
    arguments.statistics[stat::kKeypoints] = arguments.count;
    timer.lap(stat::kTotalMicroseconds);
    return {};
}

}  // namespace benzene::models::loma
