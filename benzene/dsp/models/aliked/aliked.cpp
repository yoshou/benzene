#include "dsp/models/aliked/aliked.h"

#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#include <HAP_perf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>

#include "dsp/hvx/dot_products.h"
#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/prefetch.h"
#include "dsp/hvx/scalar_math.h"
#include "dsp/hvx/transpose.h"
#include "dsp/hvx/vector.h"
#include "dsp/kernels/convolution_int8/convolution_int8.h"
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

namespace benzene::models::aliked {
namespace {

namespace tensor = protocol::aliked::tensor;
namespace stat = protocol::aliked::stat;
namespace convolution_int8 = kernels::convolution_int8;
using hvx::Mask;
using hvx::Vector;
using hvx::VectorPair;
using SeluVectors = vector_math::SeluConstants;

constexpr int kMaximumWidth = 2048;
constexpr int kMaximumDescriptorHeight = 4096;
constexpr size_t kSlack = 64 * 1024;  // past every buffer: kernels read and write whole groups, gather past a row's end
constexpr size_t kDeformableLaneTableBytes = 96 * 128;
constexpr size_t kDeformableTableBytes = 32 * 96 * 4;
constexpr int kDescriptorScratchVectors = 64 * 4;
constexpr int kSortSlices = 4;

// protocol::aliked::tensor::kBlock1Conv1 (3x3, 1 -> 16 channels, input zero point 0), in this order:
// - words [16 channels][9 taps] holding the int16 weight in both halfwords (the vmpy operand of even and odd pixels);
// - the SELU constants [16 channels][convolution_int8::kSeluConstantCount].
struct Block1Conv1Tensor {
    static constexpr int kChannels = 16;
    static constexpr size_t kWeightBytes = hvx::round_up<size_t>(kChannels * 9 * 4, hvx::kVectorBytes);
    const int32_t * weights;
    const float * constants;
    explicit Block1Conv1Tensor(const uint8_t * tensor)
        : weights(reinterpret_cast<const int32_t *>(tensor)),
          constants(reinterpret_cast<const float *>(tensor + kWeightBytes)) {}
};

// A score head 3x3 layer (protocol::aliked::tensor::kScoreHead2 / 4 / 6), its input channels in groups of 4 (score1: 2
// groups, score2 / score3: 1). In this order:
// - the int8 weights: words [output channel][9 taps][groups] (the group's 4 input channels' weights, the vrmpy
//   operand);
// - per output channel the int32 correction sum(w z) over all taps and input channels (the sums are taken over the
//   raw codes; padding pixels read z);
// - the output constants: [output channel][10] SELU constants (score_head.2, .4), or S, B (score_head.6: the logit =
//   f32(f32(acc) S) + B).
struct ScoreHeadTensor {
    const int32_t * weights = nullptr;
    const int32_t * corrections = nullptr;
    const float * constants = nullptr;
    ScoreHeadTensor() = default;
    ScoreHeadTensor(const uint8_t * tensor, int output_channels, int input_groups)
        : weights(reinterpret_cast<const int32_t *>(tensor)),
          corrections(reinterpret_cast<const int32_t *>(tensor + weight_bytes(output_channels, input_groups))),
          constants(reinterpret_cast<const float *>(tensor + weight_bytes(output_channels, input_groups) +
                                                    hvx::round_up<size_t>(output_channels * 4, hvx::kVectorBytes))) {}
    static constexpr size_t weight_bytes(int output_channels, int input_groups) {
        return hvx::round_up<size_t>(output_channels * 9 * input_groups * 4, hvx::kVectorBytes);
    }
};

// protocol::aliked::tensor::kStageConstants: fp32 vectors of 32 lanes (128 bytes each), in this order:
// 1. pool K, Z per pooled tensor: x1.pool (2 x 2 of x1: channels 0 .. 31, the 16 beyond 15 give 0), x2.pool (4 x 4
//    of x2), x3.pool (4 x 4 of x3: 2 vectors each). q = trunc(f32(f32(sum) K) + Z).
// 2. aggregation scale, zero per agg1 .. agg4 (32 channels): x = f32(f32(q - z) s).
// 3. score1 SELU, 10 vectors: the constants of score1's 8 channels tiled 4 times (lane = 8 p + c).
// 4. score zero points, 3 vectors: score1's 8 zero points tiled 4 times, score2's / score3's 4 tiled 8 times (as fp32
//    values).
// 5. patch R, Z: desc.patch's quantizer (128 channels, 4 vectors each), q = trunc(f32(f32(x R) + Z)).
// 6. samples R, Z: desc.samples' quantizer.
struct StageConstants {
    const float * x1_pool_k;
    const float * x1_pool_z;
    const float * x2_pool_k;
    const float * x2_pool_z;
    const float * x3_pool_k;
    const float * x3_pool_z;
    const float * aggregation_scale[4];
    const float * aggregation_zero[4];
    const float * score1_selu;
    const float * score_zero[3];
    const float * patch_r;
    const float * patch_z;
    const float * samples_r;
    const float * samples_z;
    static constexpr size_t kVectors = 2 + 2 + 4 + 8 + 10 + 3 + 16;
    explicit StageConstants(const uint8_t * tensor) {
        const auto * v = reinterpret_cast<const float *>(tensor);
        x1_pool_k = v;
        x1_pool_z = v + 32;
        x2_pool_k = v + 64;
        x2_pool_z = v + 96;
        x3_pool_k = v + 128;
        x3_pool_z = v + 192;
        for (int k = 0; k < 4; k++) {
            aggregation_scale[k] = v + 256 + 32 * k;
            aggregation_zero[k] = v + 384 + 32 * k;
        }
        score1_selu = v + 512;
        for (int k = 0; k < 3; k++) score_zero[k] = v + 832 + 32 * k;
        patch_r = v + 928;
        patch_z = v + 1056;
        samples_r = v + 1184;
        samples_z = v + 1312;
    }
};
static_assert(StageConstants::kVectors * 32 == 1440);

// The taps of an align_corners resampling from `source` to `target` samples. Per target index i:
// src = f32(f32(source - 1) / f32(target - 1)) i, first = trunc(src), second = min(first + 1, source - 1),
// second_weight = src - first, first_weight = 1 - second_weight.
struct UpsampleTaps {
    int * first;
    int * second;
    float * first_weight;
    float * second_weight;
};

// A sample's nw corner pixel and the weights of its corners nw, ne, sw, se.
struct SamplePosition {
    int y;
    int x;
    float weight[4];
};

inline SeluVectors splat_selu_constants(const float * constants) {
    SeluVectors c;
    c.positive_scale = Vector<float>::splat(constants[0]);
    c.positive_offset = Vector<float>::splat(constants[1]);
    c.exponent_scale = Vector<float>::splat(constants[2]);
    c.exponent_offset = Vector<float>::splat(constants[3]);
    c.negative_offset = Vector<float>::splat(constants[4]);
    for (int i = 0; i < 5; i++) c.exponential[i] = Vector<float>::splat(constants[5 + i]);
    return c;
}

inline SeluVectors lane_selu_constants(const float * vectors) {
    const auto lanes = [&](int k) { return hvx::load_unaligned(vectors + 32 * k); };
    SeluVectors c;
    c.positive_scale = lanes(0);
    c.positive_offset = lanes(1);
    c.exponent_scale = lanes(2);
    c.exponent_offset = lanes(3);
    c.negative_offset = lanes(4);
    for (int i = 0; i < 5; i++) c.exponential[i] = lanes(5 + i);
    return c;
}

inline Vector<int32_t> selu_codes(Vector<int32_t> accumulators, const SeluVectors & constants) {
    const Vector<float> y = vector_math::selu_requantize(to_float(accumulators), constants);
    return truncate_to_int32(y);
}

// For fp32 pre-activations: the constants of S = 1, B = 0.
inline Vector<int32_t> selu_codes_of_values(Vector<float> values, const SeluVectors & constants) {
    return truncate_to_int32(vector_math::selu_requantize(values, constants));
}

// bytes [32 k, 32 k + 32) = vector k's lanes, saturated to 0 .. 255.
inline Vector<uint8_t> pack_codes(Vector<int32_t> v0, Vector<int32_t> v1, Vector<int32_t> v2, Vector<int32_t> v3) {
    return pack_to_uint8_saturating(pack_to_int16_saturating(v3, v2), pack_to_int16_saturating(v1, v0));
}

constexpr int kChannels = Block1Conv1Tensor::kChannels;

// - image: the gray image with a zero row above and below and zero bytes left (128) and right of each row. Pixel
//   (y, x) is at image + y * row_bytes + x for y = -1 .. height and x = -128 .. W + 127, W the width rounded up to a
//   multiple of 128.
// - output: [height][width][32] codes, channels 16 .. 31 zero.
// - tensor: a Block1Conv1Tensor.
// Width a multiple of 4, height >= 1.
struct Block1Conv1Arguments {
    const uint8_t * image;
    size_t row_bytes;
    int height;
    int width;
    const uint8_t * tensor;
    uint8_t * output;
};

struct Block1Job {
    const Block1Conv1Arguments * arguments;
    const int32_t * weights;
    const SeluVectors * constants;
};

struct ChannelConstants {
    alignas(128) SeluVectors channels[kChannels];
};

// 32 pixels' 16 channel codes (four word vectors, one per channel quad) -> 4 vectors of 8 pixels x 16 bytes.
inline void interleave_pixel_words(const Vector<uint8_t> words[4], Vector<uint8_t> out[4]) {
    const VectorPair<uint8_t> quads01 = interleave_chunks<4>(words[1], words[0]);
    const VectorPair<uint8_t> quads23 = interleave_chunks<4>(words[3], words[2]);
    const VectorPair<uint8_t> low = interleave_chunks<8>(quads23.low(), quads01.low());
    const VectorPair<uint8_t> high = interleave_chunks<8>(quads23.high(), quads01.high());
    out[0] = low.low();
    out[1] = low.high();
    out[2] = high.low();
    out[3] = high.high();
}

void block1_conv1_row(void * context, int y, int) {
    const auto * job = static_cast<const Block1Job *>(context);
    const Block1Conv1Arguments & a = *job->arguments;
    const uint8_t * rows[3];
    for (int dy = 0; dy < 3; dy++) rows[dy] = a.image + static_cast<ptrdiff_t>(y - 1 + dy) * a.row_bytes;
    uint8_t * output_row = a.output + static_cast<size_t>(y) * a.width * 32;
    const Vector<int32_t> zero = Vector<int32_t>::zero();

    for (int x0 = 0; x0 < a.width; x0 += 128) {
        VectorPair<uint16_t> taps[9];
        for (int dy = 0; dy < 3; dy++) {
            const auto * row = hvx::vectors(rows[dy] + x0);
            const Vector<uint8_t> current = row[0];
            taps[3 * dy] = widen(slide_up(current, row[-1], 1));
            taps[3 * dy + 1] = widen(current);
            taps[3 * dy + 2] = widen(slide_down(row[1], current, 1));
        }

        for (int half = 0; half < 2; half++) {
            const int first_pixel = x0 + 64 * half;
            if (first_pixel >= a.width) break;

            Vector<uint8_t> even_words[4];
            Vector<uint8_t> odd_words[4];
            for (int quad = 0; quad < 4; quad++) {
                Vector<int32_t> even[4];  // lane i: pixel first_pixel + 2 i (odd: + 2 i + 1)
                Vector<int32_t> odd[4];
#pragma clang loop unroll(full)
                for (int c = 0; c < 4; c++) {
                    const int channel = 4 * quad + c;
                    VectorPair<int32_t> sums = make_pair(zero, zero);
                    const int32_t * weights = job->weights + 9 * channel;
#pragma clang loop unroll(full)
                    for (int tap = 0; tap < 9; tap++) {
                        const Vector<int16_t> pixels =
                            reinterpret<int16_t>(half != 0 ? taps[tap].high() : taps[tap].low());
                        sums = multiply_accumulate_widening(sums, pixels, weights[tap]);
                    }
                    even[c] = selu_codes(sums.low(), job->constants[channel]);
                    odd[c] = selu_codes(sums.high(), job->constants[channel]);
                }

                even_words[quad] = pack_to_uint8_interleaved(even[0], even[1], even[2], even[3]);
                odd_words[quad] = pack_to_uint8_interleaved(odd[0], odd[1], odd[2], odd[3]);
            }

            Vector<uint8_t> even_pixels[4];
            Vector<uint8_t> odd_pixels[4];
            interleave_pixel_words(even_words, even_pixels);
            interleave_pixel_words(odd_words, odd_pixels);

            auto * destination = hvx::vectors(output_row + static_cast<size_t>(first_pixel) * 32);
            for (int k = 0; k < 4; k++) {
                const VectorPair<uint8_t> pixels = interleave_chunks<16>(odd_pixels[k], even_pixels[k]);
                for (int part = 0; part < 2; part++) {
                    const VectorPair<uint8_t> padded =
                        interleave_chunks<16>(reinterpret<uint8_t>(zero), part != 0 ? pixels.high() : pixels.low());
                    const int pixel = first_pixel + 16 * k + 8 * part;
                    if (pixel < a.width) destination[4 * k + 2 * part] = padded.low();
                    if (pixel + 4 < a.width) destination[4 * k + 2 * part + 1] = padded.high();
                }
            }
        }
    }
}

runtime::Status run_block1_conv1(runtime::Session & session, const Block1Conv1Arguments & arguments) {
    ChannelConstants * constants = session.state<ChannelConstants>();
    if (constants == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const Block1Conv1Tensor tensor(arguments.tensor);
    for (int channel = 0; channel < kChannels; channel++) {
        constants->channels[channel] =
            splat_selu_constants(tensor.constants + convolution_int8::kSeluConstantCount * channel);
    }
    Block1Job job{&arguments, tensor.weights, constants->channels};
    return runtime::parallel_for(session, runtime::kHvxThreads, arguments.height, block1_conv1_row, &job);
}

constexpr int kMaximumPooledWidth = 2048 / 32;

// input [height][width][channels] -> output [height / window][width / window][channels], per channel
// q = clamp(trunc(f32(f32(f32(sum) K) + Z)), 0, 255) of the window's sum, with
// - 32 channels: window 2 or 4, output widths multiples of 4;
// - 64 channels: window 4, output width at most 64.
// upper_half (optional, 32 channels only): the same codes again as channels 32 .. 63 of a map of 64 channels.
struct PoolingArguments {
    const uint8_t * input;
    int height;
    int width;
    int channels;
    int window;
    const float * multipliers;  // K [channels]
    const float * offsets;      // Z [channels]
    uint8_t * output;
    uint8_t * upper_half;
};

inline Vector<int32_t> requantize(Vector<int32_t> sums, Vector<float> multipliers, Vector<float> offsets) {
    return truncate_to_int32(to_float(sums) * multipliers + offsets);
}

inline Vector<int32_t> low_lanes_as_words(Vector<int16_t> halfwords) {
    return reinterpret<int32_t>(widen(reinterpret<uint16_t>(halfwords)).low());
}

inline VectorPair<int16_t> row_sum(const uint8_t * first, size_t row_bytes, int rows) {
    VectorPair<int16_t> sum = reinterpret<int16_t>(widen(hvx::load(first)));
    for (int r = 1; r < rows; r++) {
        const VectorPair<int16_t> next = reinterpret<int16_t>(widen(hvx::load(first + r * row_bytes)));
        sum = make_pair(sum.high() + next.high(), sum.low() + next.low());
    }
    return sum;
}

// A halfword vector holds two pixels of 32 channels: lanes i + 32 added onto lanes i.
inline Vector<int16_t> add_upper_pixel(Vector<int16_t> halfwords) {
    return halfwords + rotate_right_bytes(halfwords, 64);
}

template <int Window>
void pool32_row(const PoolingArguments * a, const uint8_t * rows, size_t row_bytes, int y, Vector<float> k0,
                Vector<float> z0) {
    const int output_width = a->width / Window;
    uint8_t * output = a->output + static_cast<size_t>(y) * output_width * 32;
    uint8_t * upper = a->upper_half != nullptr ? a->upper_half + static_cast<size_t>(y) * output_width * 64 : nullptr;
    // bytes 32 .. 63 and 96 .. 127
    const Mask upper_halves = and_not(Mask::first_bytes(64), Mask::first_bytes(32)) | !Mask::first_bytes(96);

    for (int x = 0; x < output_width; x += 4) {
        Vector<int32_t> codes[4];
        if (Window == 2) {
#pragma clang loop unroll(full)
            for (int pair = 0; pair < 2; pair++) {
                const VectorPair<int16_t> sums =
                    row_sum(rows + static_cast<size_t>(2 * x + 4 * pair) * 32, row_bytes, 2);
                codes[2 * pair] = requantize(low_lanes_as_words(add_upper_pixel(sums.low())), k0, z0);
                codes[2 * pair + 1] = requantize(low_lanes_as_words(add_upper_pixel(sums.high())), k0, z0);
            }
        } else {
#pragma clang loop unroll(full)
            for (int p = 0; p < 4; p++) {
                const VectorPair<int16_t> sums = row_sum(rows + static_cast<size_t>(4 * (x + p)) * 32, row_bytes, 4);
                const Vector<int16_t> total = add_upper_pixel(sums.low() + sums.high());
                codes[p] = requantize(low_lanes_as_words(total), k0, z0);
            }
        }

        const Vector<uint8_t> packed = pack_codes(codes[0], codes[1], codes[2], codes[3]);
        store(output + static_cast<size_t>(x) * 32, packed);
        if (upper != nullptr) {
            const VectorPair<uint8_t> spread = interleave_chunks<32>(packed, packed);
            store_masked(upper + static_cast<size_t>(x) * 64, upper_halves, spread.low());
            store_masked(upper + static_cast<size_t>(x) * 64 + 128, upper_halves, spread.high());
        }
    }
}

void pool_row(void * context, int y, int) {
    const auto * a = static_cast<const PoolingArguments *>(context);
    const int output_width = a->width / a->window;
    const size_t row_bytes = static_cast<size_t>(a->width) * a->channels;
    const uint8_t * rows = a->input + static_cast<size_t>(y) * a->window * row_bytes;
    const Vector<float> k0 = hvx::load_unaligned(a->multipliers);
    const Vector<float> z0 = hvx::load_unaligned(a->offsets);

    // 64 channels: the row is assembled in an aligned buffer, as rows of odd output widths are not 128-byte aligned.
    if (a->channels == 64) {
        const Vector<float> k1 = hvx::load_unaligned(a->multipliers + 32);
        const Vector<float> z1 = hvx::load_unaligned(a->offsets + 32);
        alignas(128) uint8_t row_codes[kMaximumPooledWidth * 64 + 128];

        for (int x = 0; x < output_width; x += 2) {
            Vector<int32_t> codes[4];
            for (int p = 0; p < 2; p++) {
                const int pixel = x + p < output_width ? x + p : x;
                const uint8_t * first = rows + static_cast<size_t>(pixel) * 4 * 64;
                const VectorPair<int16_t> a01 = row_sum(first, row_bytes, 4);
                const VectorPair<int16_t> a23 = row_sum(first + 128, row_bytes, 4);
                const Vector<int16_t> total = a01.low() + a01.high() + (a23.low() + a23.high());
                const VectorPair<uint32_t> words = widen(reinterpret<uint16_t>(total));
                codes[2 * p] = requantize(reinterpret<int32_t>(words.low()), k0, z0);
                codes[2 * p + 1] = requantize(reinterpret<int32_t>(words.high()), k1, z1);
            }
            store(row_codes + static_cast<size_t>(x) * 64, pack_codes(codes[0], codes[1], codes[2], codes[3]));
        }

        memcpy(a->output + static_cast<size_t>(y) * output_width * 64, row_codes,
               static_cast<size_t>(output_width) * 64);
        return;
    }

    if (a->window == 2) {
        pool32_row<2>(a, rows, row_bytes, y, k0, z0);
    } else {
        pool32_row<4>(a, rows, row_bytes, y, k0, z0);
    }
}

runtime::Status run_average_pool(runtime::Session & session, const PoolingArguments & arguments) {
    return runtime::parallel_for(session, runtime::kHvxThreads, arguments.height / arguments.window, pool_row,
                                 const_cast<PoolingArguments *>(&arguments));
}

constexpr int kDequantizeItemPixels = 64;
constexpr int kItemPixels = 32;
// A pixel's table row: tap k's four corner weights at words 8k .. 8k + 3, their pixel indices at 8k + 4 .. 8k + 7;
// padded to 3 blocks of 32 words.
constexpr int kPixelRowWords = 96;

// The columns of a 3x3 torchvision deform_conv2d (DCNv1): the input sampled at the offset positions, as codes.
// - input: [height][width][channels] codes (channels 32, 64 or 128) with per-channel zero points `zero` (fp32
//   values).
// - offsets: fp32 rows [height * width][32] (channel 2k: dy of tap k = 3 ky + kx, 2k + 1: dx), clamped here to
//   +-max(height, width) / 4.
// - columns: [height * width][column_stride] codes, tap-major (bytes [tap * channels, + channels)). With residual
//   (optional), the residual map's residual_channels codes per pixel follow the 9 taps. The rest of each row up to
//   column_stride is zero.
struct DeformableArguments {
    const uint8_t * input;
    int height;
    int width;
    int channels;
    const float * zero;
    const float * offsets;
    float * dequantized;  // workspace: [height * width + 1][channels] fp32
    uint8_t * tables;     // workspace: per thread kDeformableLaneTableBytes, then per 32 pixels kDeformableTableBytes
    const uint8_t * residual;
    int residual_channels;
    uint8_t * columns;
    int column_stride;
};

struct SamplingJob {
    const DeformableArguments * arguments;
    Vector<int32_t> zero_codes[4];
    Vector<float> rounding[4];
};

void dequantize_pixels(void * context, int item, int) {
    const auto * job = static_cast<const SamplingJob *>(context);
    const DeformableArguments & a = *job->arguments;
    const int chunks = a.channels / 32;
    const int pixels = a.height * a.width;
    const int first = kDequantizeItemPixels * item;

    for (int p = first; p < first + kDequantizeItemPixels && p <= pixels; p++) {
        auto * out = hvx::vectors(a.dequantized + static_cast<size_t>(p) * a.channels);
        if (p == pixels) {
            for (int chunk = 0; chunk < chunks; chunk++) out[chunk] = Vector<float>::zero();
            continue;
        }

        const uint8_t * codes = a.input + static_cast<size_t>(p) * a.channels;
        for (int chunk = 0; chunk < chunks; chunk++) {
            const VectorPair<uint16_t> halfwords = widen(hvx::load_unaligned(codes + 32 * chunk));
            const Vector<int32_t> words = reinterpret<int32_t>(widen(halfwords.low()).low());
            out[chunk] = to_float(words - job->zero_codes[chunk]);
        }
    }
}

// A tap's four corner weights and pixel indices of 32 pixels, lane = pixel.
struct TapTable {
    Vector<float> weights[4];
    Vector<int32_t> corners[4];
};

// deform_conv2d's bilinear sampling, operation for operation; floor(v) = trunc(v) - 1 where v < trunc(v).
void position_table(const DeformableArguments & a, const float * offsets, HVX_Vector rows, HVX_Vector columns,
                    TapTable * table) {
    Vector<uint32_t> offset_columns[32];
    hvx::transpose_words_32x32_in_registers(offsets, 128, offset_columns, offset_columns + 16);

    const float limit = static_cast<float>(a.height > a.width ? a.height : a.width) / 4.0f;
    const Vector<float> upper = Vector<float>::splat(limit);
    const Vector<float> lower = Vector<float>::splat(-limit);
    const Vector<float> minus_one = Vector<float>::splat(-1.0f);
    const Vector<float> one = Vector<float>::splat(1.0f);
    const Vector<float> height = Vector<float>::splat(static_cast<float>(a.height));
    const Vector<float> width = Vector<float>::splat(static_cast<float>(a.width));
    const Vector<float> rows_f32 = to_float(Vector<int32_t>{rows});
    const Vector<float> columns_f32 = to_float(Vector<int32_t>{columns});
    const Vector<int32_t> zero_pixel = Vector<int32_t>::splat(a.height * a.width);
    const Vector<int32_t> last_row = Vector<int32_t>::splat(a.height - 1);
    const Vector<int32_t> last_column = Vector<int32_t>::splat(a.width - 1);
    const Vector<int32_t> minus_one_int = Vector<int32_t>::splat(-1);
    // in both halfwords: multiply_by_halfwords multiplies even words by the low one, odd words by the high one
    const int width_halfwords = (a.width << 16) | a.width;

#pragma clang loop unroll(full)
    for (int tap = 0; tap < 9; tap++) {
        const Vector<float> dy = min(max(reinterpret<float>(offset_columns[2 * tap]), lower), upper);
        const Vector<float> dx = min(max(reinterpret<float>(offset_columns[2 * tap + 1]), lower), upper);
        const Vector<float> base_y = rows_f32 + static_cast<float>(tap / 3 - 1);
        const Vector<float> base_x = columns_f32 + static_cast<float>(tap % 3 - 1);
        const Vector<float> py = base_y + dy;
        const Vector<float> px = base_x + dx;
        const Mask inside = ((py > minus_one) & (height > py)) & ((px > minus_one) & (width > px));

        const Vector<int32_t> truncated_y = truncate_to_int32(py);
        const Vector<int32_t> truncated_x = truncate_to_int32(px);
        const Vector<int32_t> iy = select(to_float(truncated_y) > py, truncated_y + minus_one_int, truncated_y);
        const Vector<int32_t> ix = select(to_float(truncated_x) > px, truncated_x + minus_one_int, truncated_x);

        const Vector<float> ly = py - to_float(iy);
        const Vector<float> lx = px - to_float(ix);
        const Vector<float> hy = one - ly;
        const Vector<float> hx = one - lx;
        const Vector<float> weights[4] = {hy * hx, hy * lx, ly * hx, ly * lx};

        const Vector<int32_t> next_y = iy - minus_one_int;
        const Vector<int32_t> next_x = ix - minus_one_int;
        const Mask row_valid[2] = {and_not(iy > minus_one_int, iy > last_row),
                                   and_not(next_y > minus_one_int, next_y > last_row)};
        const Mask column_valid[2] = {and_not(ix > minus_one_int, ix > last_column),
                                      and_not(next_x > minus_one_int, next_x > last_column)};

        const Vector<int32_t> row_start[2] = {multiply_by_halfwords(iy, width_halfwords),
                                              multiply_by_halfwords(next_y, width_halfwords)};
        const Vector<int32_t> column[2] = {ix, next_x};

        TapTable & entry = table[tap];
#pragma clang loop unroll(full)
        for (int c = 0; c < 4; c++) {
            entry.weights[c] = select(inside, weights[c], Vector<float>::zero());
            const Mask valid = inside & (row_valid[c / 2] & column_valid[c % 2]);
            entry.corners[c] = select(valid, row_start[c / 2] + column[c % 2], zero_pixel);
        }
    }
}

// Summed in deform_conv2d's order: nw, ne, sw, se.
inline Vector<int32_t> sample_chunk(const Vector<float> * const corners[4], const Vector<float> weights[4], int chunk,
                                    Vector<float> rounding) {
    Vector<float> value = weights[0] * corners[0][chunk];
    value = value + weights[1] * corners[1][chunk];
    value = value + weights[2] * corners[2][chunk];
    value = value + weights[3] * corners[3][chunk];
    return truncate_to_int32(value + rounding);
}

template <int Chunks, int ResidualChunks>
[[gnu::always_inline]] inline void sample_pixel(const float * __restrict weights_row,
                                                const int32_t * __restrict corners_row,
                                                const Vector<float> * __restrict pixels, const Vector<float> * rounding,
                                                const uint8_t * __restrict residual,
                                                Vector<uint8_t> * __restrict columns) {
    constexpr int kCodes = 9 * Chunks + ResidualChunks;
    constexpr int kVectors = (kCodes + 3) / 4;
    Vector<int32_t> codes[4 * kVectors];

#pragma clang loop unroll(full)
    for (int tap = 0; tap < 9; tap++) {
        const Vector<float> * corners[4];
        Vector<float> weights[4];
#pragma clang loop unroll(full)
        for (int c = 0; c < 4; c++) {
            weights[c] = Vector<float>::splat(weights_row[8 * tap + c]);
            corners[c] = pixels + static_cast<size_t>(corners_row[8 * tap + 4 + c]) * Chunks;
        }

#pragma clang loop unroll(full)
        for (int chunk = 0; chunk < Chunks; chunk++) {
            const int index = tap * Chunks + chunk;
            codes[index] = sample_chunk(corners, weights, chunk, rounding[chunk]);
            if (index % 4 == 3) {
                columns[index / 4].bits =
                    pack_codes(codes[index - 3], codes[index - 2], codes[index - 1], codes[index]).bits;
            }
        }
    }

#pragma clang loop unroll(full)
    for (int index = 9 * Chunks; index < 4 * kVectors; index++) {
        const int chunk = index - 9 * Chunks;
        if (chunk < ResidualChunks) {
            const VectorPair<uint16_t> halfwords = widen(hvx::load_unaligned(residual + 32 * chunk));
            codes[index] = reinterpret<int32_t>(widen(halfwords.low()).low());
        } else {
            codes[index] = Vector<int32_t>::zero();
        }
        if (index % 4 == 3) {
            columns[index / 4].bits =
                pack_codes(codes[index - 3], codes[index - 2], codes[index - 1], codes[index]).bits;
        }
    }
}

uint8_t * item_table(const DeformableArguments & a, int item) {
    return a.tables + kDeformableLaneTableBytes * runtime::kHvxThreads +
           static_cast<size_t>(item) * kDeformableTableBytes;
}

void item_positions(void * context, int item, int thread) {
    const auto * job = static_cast<const SamplingJob *>(context);
    const DeformableArguments & a = *job->arguments;
    const int first = kItemPixels * item;

    alignas(128) int32_t lane_rows[kItemPixels];
    alignas(128) int32_t lane_columns[kItemPixels];
    for (int lane = 0, y = first / a.width, x = first % a.width; lane < kItemPixels; lane++) {
        lane_rows[lane] = y;
        lane_columns[lane] = x;
        if (++x == a.width) {
            x = 0;
            y++;
        }
    }

    auto * lanes = reinterpret_cast<TapTable *>(a.tables + static_cast<size_t>(thread) * kDeformableLaneTableBytes);
    position_table(a, a.offsets + static_cast<size_t>(first) * 32, hvx::load(lane_rows).bits,
                   hvx::load(lane_columns).bits, lanes);

    auto * rows = reinterpret_cast<Vector<uint32_t> *>(item_table(a, item));
    for (int block = 0; block < kPixelRowWords / 32; block++) {
        Vector<uint32_t> columns[32];
        hvx::transpose_words_32x32_in_registers(lanes + 4 * block, 128, columns, columns + 16);
        for (int pixel = 0; pixel < 32; pixel++) rows[pixel * (kPixelRowWords / 32) + block] = columns[pixel];
    }
}

template <int Chunks, int ResidualChunks>
void sample_pixel_columns(void * context, int item, int) {
    const auto * job = static_cast<const SamplingJob *>(context);
    const DeformableArguments & a = *job->arguments;
    const int first = kItemPixels * item;
    const int end = first + kItemPixels < a.height * a.width ? first + kItemPixels : a.height * a.width;
    const uint8_t * table = item_table(a, item);

    Vector<float> rounding[Chunks];
#pragma clang loop unroll(full)
    for (int chunk = 0; chunk < Chunks; chunk++) rounding[chunk] = job->rounding[chunk];

    for (int p = first; p < end; p++) {
        const uint8_t * row = table + static_cast<size_t>(p - first) * kPixelRowWords * 4;
        for (int line = 0; line < kPixelRowWords * 4; line += 32) {
            hvx::prefetch_to_l1(row + kPixelRowWords * 4 + line);
        }
        sample_pixel<Chunks, ResidualChunks>(reinterpret_cast<const float *>(row),
                                             reinterpret_cast<const int32_t *>(row), hvx::vectors(a.dequantized),
                                             rounding, a.residual + static_cast<size_t>(p) * (32 * ResidualChunks),
                                             hvx::vectors(a.columns + static_cast<size_t>(p) * a.column_stride));
    }
}

runtime::Status run_deformable_sampling(runtime::Session & session, const DeformableArguments & arguments) {
    static_assert(kPixelRowWords * 4 * kItemPixels <= kDeformableTableBytes, "deformable table");
    static_assert(kPixelRowWords * 128 <= kDeformableLaneTableBytes, "deformable lane table");

    const int residual_chunks = arguments.residual != nullptr ? arguments.residual_channels / 32 : 0;
    runtime::ItemFunction sample = nullptr;
    if (arguments.channels == 32 && residual_chunks == 0) sample = sample_pixel_columns<1, 0>;
    if (arguments.channels == 64 && residual_chunks == 1) sample = sample_pixel_columns<2, 1>;
    if (arguments.channels == 64 && residual_chunks == 0) sample = sample_pixel_columns<2, 0>;
    if (arguments.channels == 128 && residual_chunks == 2) sample = sample_pixel_columns<4, 2>;
    const int codes = 9 * (arguments.channels / 32) + residual_chunks;
    if (sample == nullptr || arguments.column_stride != (codes + 3) / 4 * 128) {
        return std::unexpected(runtime::Error::kBadParameter);
    }

    alignas(128) SamplingJob job;
    job.arguments = &arguments;
    for (int chunk = 0; chunk < arguments.channels / 32; chunk++) {
        alignas(128) int32_t zero_codes[32];
        alignas(128) float rounding[32];
        for (int lane = 0; lane < 32; lane++) {
            const float z = arguments.zero[32 * chunk + lane];
            zero_codes[lane] = static_cast<int32_t>(z);
            rounding[lane] = z + 0.5f;
        }
        job.zero_codes[chunk] = hvx::load(zero_codes);
        job.rounding[chunk] = hvx::load(rounding);
    }

    const int pixels = arguments.height * arguments.width;
    // the map's pixels and the zero pixel after them
    runtime::Status status = runtime::parallel_for(session, runtime::kHvxThreads, pixels / kDequantizeItemPixels + 1,
                                                   dequantize_pixels, &job);
    if (!status) return status;

    const int items = (pixels + kItemPixels - 1) / kItemPixels;
    status = runtime::parallel_for(session, runtime::kHvxThreads, items, item_positions, &job);
    if (!status) return status;
    return runtime::parallel_for(session, runtime::kHvxThreads, items, sample, &job);
}

constexpr int kPadPixels = 32;
constexpr int kHalo = 3;

// The ring rows of one thread: 3 rows of score1 (two planes of 4 channels), score2 and score3 (one plane each), each
// plane row (width + 64) pixels of 4 bytes.
constexpr int score_band_buffer_bytes(int width) {
    return 12 * (width + 64) * 4;
}

constexpr size_t weight_lanes_bytes(int width) {
    return static_cast<size_t>(3) * width * 64;
}
constexpr size_t thread_vtcm_bytes(int width) {
    return score_band_buffer_bytes(width) + static_cast<size_t>(6) * width * 32 + static_cast<size_t>(width) * 16 + 128;
}

constexpr int kFactors[4] = {1, 2, 8, 32};

// - parts[k]: score_head.0's 1x1 over agg_{k+1}, fp32 rows [height / f][width / f][8], f = 1, 2, 8, 32.
// - column_taps[k] / row_taps[k]: the upsampling taps of part k (k = 1 .. 3; [0] unused).
// - layer_tensors: score_head.2 / .4 / .6, ScoreHeadTensor layouts.
// - score_map: [height][width] fp32.
// - vtcm: score_head_vtcm_bytes(width) of VTCM.
struct ScoreHeadArguments {
    int height;
    int width;
    const float * parts[4];
    UpsampleTaps column_taps[4];
    const float * column_weight_lanes[4];  // compute_column_weight_lanes of column_taps[k]
    UpsampleTaps row_taps[4];
    const uint8_t * layer_tensors[3];
    const float * score1_selu;    // [10][32]: 8 channels tiled 4 times
    const float * score_zero[3];  // score1 (8 x 4), score2 (4 x 8), score3 (4 x 8) as fp32
    float * score_map;
    uint8_t * vtcm;
    int32_t * statistics;
};

// 32 bytes of each of four sources at bytes [32 j, 32 j + 32) of one vector.
inline Vector<float> gather_four_pixels(const float * a, const float * b, const float * c, const float * d) {
    const Vector<float> v0 = hvx::load_unaligned(a);
    const Vector<float> v1 = rotate_right_bytes(hvx::load_unaligned(b), 96);
    const Vector<float> v2 = rotate_right_bytes(hvx::load_unaligned(c), 64);
    const Vector<float> v3 = rotate_right_bytes(hvx::load_unaligned(d), 32);
    const Vector<float> low = select(Mask::first_bytes(32), v0, v1);
    const Vector<float> high = select(Mask::first_bytes(96), v2, v3);
    return select(Mask::first_bytes(64), low, high);
}

// horizontal[k][r & 1] holds row r of part k upsampled horizontally: an output row's two source rows are adjacent.
struct ThreadRows {
    uint8_t * rings;
    float * horizontal[4][2];
    int horizontal_source[4][2];
    float * staging;
    uint64_t upsampling_microseconds;
};

struct ScoreJob {
    const ScoreHeadArguments * arguments;
    const Vector<float> * weight_lanes[4];
    uint64_t upsampling_microseconds[runtime::kHvxThreads];
    SeluVectors score1;
    Vector<uint8_t> pads[4];
    SeluVectors layer_constants[2][4];
    ScoreHeadTensor layers[3];
    int bands;
};

// A plane row: pixel x's word (the codes of 4 channels) at word 32 + x. The 32 pixels on either side, and the rows
// outside the image, hold the zero points.
struct PlaneRows {
    uint8_t * base;
    int row_bytes;
    uint8_t * row(int ring_index, int plane) const {
        return base + static_cast<size_t>(ring_index * 2 + plane) * row_bytes;
    }
};

inline int ring(int row) {
    return ((row % 3) + 3) % 3;
}

void fill_pad_row(uint8_t * row, int width, Vector<uint8_t> pad) {
    auto * words = hvx::vectors(row);
    for (int v = 0; v < (width + 2 * kPadPixels) / 32; v++) words[v] = pad;
}

const Vector<float> * horizontal_row(const ScoreJob * job, ThreadRows & thread, int k, int r) {
    float * row = thread.horizontal[k][r & 1];
    if (thread.horizontal_source[k][r & 1] == r) return hvx::vectors(row);

    thread.horizontal_source[k][r & 1] = r;
    const uint64_t start = HAP_perf_get_time_us();
    const ScoreHeadArguments & a = *job->arguments;
    const int source_width = a.width / kFactors[k];
    const float * source = a.parts[k] + static_cast<size_t>(r) * source_width * 8;
    auto * staging = hvx::vectors(thread.staging);
    for (int v = 0; v < (source_width * 32 + 127) / 128; v++) staging[v] = hvx::load_unaligned(source + 32 * v);

    const UpsampleTaps & taps = a.column_taps[k];
    const Vector<float> * weights = job->weight_lanes[k];
    const float * pixels = thread.staging;
    auto * out = hvx::vectors(row);
    for (int x = 0; x < a.width; x += 4) {
        const Vector<float> first = gather_four_pixels(pixels + 8 * taps.first[x], pixels + 8 * taps.first[x + 1],
                                                       pixels + 8 * taps.first[x + 2], pixels + 8 * taps.first[x + 3]);
        const Vector<float> second =
            gather_four_pixels(pixels + 8 * taps.second[x], pixels + 8 * taps.second[x + 1],
                               pixels + 8 * taps.second[x + 2], pixels + 8 * taps.second[x + 3]);
        out[x / 4] = weights[x / 2] * first + weights[x / 2 + 1] * second;
    }
    thread.upsampling_microseconds += HAP_perf_get_time_us() - start;
    return out;
}

void score1_row(const ScoreJob * job, ThreadRows & thread, int y, uint8_t * plane0, uint8_t * plane1) {
    const ScoreHeadArguments & a = *job->arguments;
    const int W = a.width;
    const auto * u1 = hvx::vectors(a.parts[0] + static_cast<size_t>(y) * W * 8);
    const Vector<float> * h[3][2];
    Vector<float> m[3][2];
    for (int k = 1; k <= 3; k++) {
        const UpsampleTaps & taps = a.row_taps[k];
        h[k - 1][0] = horizontal_row(job, thread, k, taps.first[y]);
        h[k - 1][1] = horizontal_row(job, thread, k, taps.second[y]);
        m[k - 1][0] = Vector<float>::splat(taps.first_weight[y]);
        m[k - 1][1] = Vector<float>::splat(taps.second_weight[y]);
    }

    auto * out0 = hvx::vectors(plane0 + kPadPixels * 4);
    auto * out1 = hvx::vectors(plane1 + kPadPixels * 4);
    const SeluVectors constants = job->score1;
    const auto sixteen_pixels = [&](int index) {
        Vector<int32_t> codes[4];
#pragma clang loop unroll(full)
        for (int v = 0; v < 4; v++) {
            Vector<float> sum = u1[index + v];
#pragma clang loop unroll(full)
            for (int k = 0; k < 3; k++) {
                const Vector<float> top = m[k][0] * h[k][0][index + v];
                const Vector<float> bottom = m[k][1] * h[k][1][index + v];
                sum = sum + (top + bottom);
            }
            codes[v] = selu_codes_of_values(sum, constants);
        }
        return pack_codes(codes[0], codes[1], codes[2], codes[3]);
    };

    for (int x = 0; x < W; x += 32) {
        const Vector<uint8_t> low = sixteen_pixels(x / 4);
        // A compiler barrier: the two halves' chains interleaved would spill.
        asm volatile("" ::: "memory");
        const Vector<uint8_t> high = sixteen_pixels(x / 4 + 4);
        // 8 bytes per pixel: the even words (channels 0 .. 3) to plane 0, the odd ones (4 .. 7) to plane 1
        const VectorPair<uint8_t> planes = deinterleave_chunks<4>(high, low);
        out0[x / 32] = planes.low();
        out1[x / 32] = planes.high();
    }
}

constexpr int kLayerGroups = 4;

template <int Outputs, int Planes, int Groups>
inline void layer_sums(const ScoreHeadTensor & layer, const uint8_t * const rows[3][2], int x,
                       Vector<int32_t> sums[Outputs][Groups]) {
#pragma clang loop unroll(full)
    for (int n = 0; n < Outputs; n++) {
#pragma clang loop unroll(full)
        for (int g = 0; g < Groups; g++) sums[n][g] = Vector<int32_t>::zero();
    }

#pragma clang loop unroll(full)
    for (int dy = 0; dy < 3; dy++) {
#pragma clang loop unroll(full)
        for (int plane = 0; plane < Planes; plane++) {
            const auto * words = hvx::vectors(rows[dy][plane] + static_cast<size_t>(kPadPixels + x) * 4);
            Vector<uint8_t> loaded[Groups + 2];
#pragma clang loop unroll(full)
            for (int g = 0; g < Groups + 2; g++) loaded[g] = words[g - 1];

#pragma clang loop unroll(full)
            for (int dx = 0; dx < 3; dx++) {
                Vector<uint8_t> shifted[Groups];
#pragma clang loop unroll(full)
                for (int g = 0; g < Groups; g++) {
                    shifted[g] = dx == 1   ? loaded[g + 1]
                                 : dx == 0 ? slide_up(loaded[g + 1], loaded[g], 4)
                                           : slide_down(loaded[g + 2], loaded[g + 1], 4);
                }

                const int tap = 3 * dy + dx;
#pragma clang loop unroll(full)
                for (int n = 0; n < Outputs; n++) {
                    const int32_t weight = layer.weights[(n * 9 + tap) * Planes + plane];
#pragma clang loop unroll(full)
                    for (int g = 0; g < Groups; g++)
                        sums[n][g] = dot_product_accumulate_scalar(sums[n][g], shifted[g], weight);
                }
            }
        }
    }

#pragma clang loop unroll(full)
    for (int n = 0; n < Outputs; n++) {
        const Vector<int32_t> correction = Vector<int32_t>::splat(layer.corrections[n]);
#pragma clang loop unroll(full)
        for (int g = 0; g < Groups; g++) sums[n][g] = sums[n][g] - correction;
    }
}

template <int Planes, int Groups>
inline void hidden_step(const ScoreJob * job, int index, const uint8_t * const rows[3][2], int x,
                        Vector<uint8_t> * out) {
    Vector<int32_t> sums[4][Groups];
    layer_sums<4, Planes, Groups>(job->layers[index], rows, x, sums);
#pragma clang loop unroll(full)
    for (int g = 0; g < Groups; g++) {
        Vector<int32_t> codes[4];
#pragma clang loop unroll(full)
        for (int n = 0; n < 4; n++) codes[n] = selu_codes(sums[n][g], job->layer_constants[index][n]);
        out[x / 32 + g] = pack_to_uint8_interleaved(codes[0], codes[1], codes[2], codes[3]);
    }
}

template <int Planes>
void hidden_row(const ScoreJob * job, int index, const uint8_t * const rows[3][2], uint8_t * output) {
    auto * out = hvx::vectors(output + kPadPixels * 4);
    const int width = job->arguments->width;
    int x = 0;
    for (; x + 32 * kLayerGroups <= width; x += 32 * kLayerGroups)
        hidden_step<Planes, kLayerGroups>(job, index, rows, x, out);
    for (; x < width; x += 32) hidden_step<Planes, 1>(job, index, rows, x, out);
}

template <int Groups>
inline void score_step(const ScoreJob * job, const uint8_t * const rows[3][2], int x, float * output) {
    const ScoreHeadTensor & layer = job->layers[2];
    Vector<int32_t> sums[1][Groups];
    layer_sums<1, 1, Groups>(layer, rows, x, sums);
    const Vector<float> scale = Vector<float>::splat(layer.constants[0]);
    const Vector<float> bias = Vector<float>::splat(layer.constants[1]);
    const Vector<float> minus_log2e = Vector<float>::splat(-1.4426950408889634f);
    const Vector<float> one = Vector<float>::splat(1.0f);

#pragma clang loop unroll(full)
    for (int g = 0; g < Groups; g++) {
        const Vector<float> logit = to_float(sums[0][g]) * scale + bias;
        const Vector<float> exponential = vector_math::exp2(logit * minus_log2e);
        store(output + x + 32 * g, vector_math::reciprocal(one + exponential));
    }
}

void score_row(const ScoreJob * job, const uint8_t * const rows[3][2], float * output) {
    const int width = job->arguments->width;
    int x = 0;
    for (; x + 32 * kLayerGroups <= width; x += 32 * kLayerGroups) score_step<kLayerGroups>(job, rows, x, output);
    for (; x < width; x += 32) score_step<1>(job, rows, x, output);
}

void score_band(void * context, int band, int thread) {
    const auto * job = static_cast<const ScoreJob *>(context);
    const ScoreHeadArguments & a = *job->arguments;
    const int first = a.height * band / job->bands;
    const int end = a.height * (band + 1) / job->bands;
    const int row_bytes = (a.width + 2 * kPadPixels) * 4;

    ThreadRows rows_of_thread;
    uint8_t * area = a.vtcm + weight_lanes_bytes(a.width) + static_cast<size_t>(thread) * thread_vtcm_bytes(a.width);
    uint8_t * upsampling_rows = area + score_band_buffer_bytes(a.width);
    rows_of_thread.rings = area;
    for (int k = 1; k < 4; k++) {
        for (int slot = 0; slot < 2; slot++) {
            rows_of_thread.horizontal[k][slot] =
                reinterpret_cast<float *>(upsampling_rows + static_cast<size_t>(2 * (k - 1) + slot) * a.width * 32);
            rows_of_thread.horizontal_source[k][slot] = -1;
        }
    }
    rows_of_thread.staging = reinterpret_cast<float *>(upsampling_rows + static_cast<size_t>(6) * a.width * 32);
    rows_of_thread.upsampling_microseconds = 0;

    const PlaneRows level1{rows_of_thread.rings, row_bytes};
    uint8_t * level2_rows = rows_of_thread.rings + 6 * row_bytes;
    uint8_t * level3_rows = rows_of_thread.rings + 9 * row_bytes;
    const auto score2_row = [&](int y) { return level2_rows + static_cast<size_t>(ring(y)) * row_bytes; };
    const auto score3_row = [&](int y) { return level3_rows + static_cast<size_t>(ring(y)) * row_bytes; };

    // The padding pixels once: the rows inside the image write only their pixels.
    for (int r = 0; r < 3; r++) {
        fill_pad_row(level1.row(r, 0), a.width, job->pads[0]);
        fill_pad_row(level1.row(r, 1), a.width, job->pads[1]);
        fill_pad_row(score2_row(r), a.width, job->pads[2]);
        fill_pad_row(score3_row(r), a.width, job->pads[3]);
    }

    for (int t = first - kHalo; t < end + kHalo; t++) {
        if (t >= 0 && t < a.height) {
            score1_row(job, rows_of_thread, t, level1.row(ring(t), 0), level1.row(ring(t), 1));
        } else {
            fill_pad_row(level1.row(ring(t), 0), a.width, job->pads[0]);
            fill_pad_row(level1.row(ring(t), 1), a.width, job->pads[1]);
        }

        const int y2 = t - 1;
        if (y2 >= first - 2 && y2 < end + 2) {
            if (y2 >= 0 && y2 < a.height) {
                const uint8_t * rows[3][2];
                for (int dy = 0; dy < 3; dy++) {
                    rows[dy][0] = level1.row(ring(y2 - 1 + dy), 0);
                    rows[dy][1] = level1.row(ring(y2 - 1 + dy), 1);
                }
                hidden_row<2>(job, 0, rows, score2_row(y2));
            } else {
                fill_pad_row(score2_row(y2), a.width, job->pads[2]);
            }
        }

        const int y3 = t - 2;
        if (y3 >= first - 1 && y3 < end + 1) {
            if (y3 >= 0 && y3 < a.height) {
                const uint8_t * rows[3][2];
                for (int dy = 0; dy < 3; dy++) {
                    rows[dy][0] = score2_row(y3 - 1 + dy);
                    rows[dy][1] = nullptr;
                }
                hidden_row<1>(job, 1, rows, score3_row(y3));
            } else {
                fill_pad_row(score3_row(y3), a.width, job->pads[3]);
            }
        }

        const int y4 = t - 3;
        if (y4 >= first && y4 < end) {
            const uint8_t * rows[3][2];
            for (int dy = 0; dy < 3; dy++) {
                rows[dy][0] = score3_row(y4 - 1 + dy);
                rows[dy][1] = nullptr;
            }
            score_row(job, rows, a.score_map + static_cast<size_t>(y4) * a.width);
        }
    }

    const_cast<ScoreJob *>(job)->upsampling_microseconds[thread] = rows_of_thread.upsampling_microseconds;
}

Vector<uint8_t> pad_words(const float * zero_points) {
    alignas(128) uint8_t bytes[128];
    for (int i = 0; i < 128; i++) bytes[i] = static_cast<uint8_t>(zero_points[i % 4]);
    return hvx::load(bytes);
}

size_t score_head_vtcm_bytes(int width) {
    return weight_lanes_bytes(width) + static_cast<size_t>(runtime::kHvxThreads) * thread_vtcm_bytes(width);
}

void compute_upsample_taps(int source, int target, const UpsampleTaps & taps) {
    const float scale = static_cast<float>(source - 1) / static_cast<float>(target - 1);
    for (int i = 0; i < target; i++) {
        const float position = scale * static_cast<float>(i);
        const int first = static_cast<int>(position);
        const float second_weight = position - static_cast<float>(first);
        taps.first[i] = first;
        taps.second[i] = first < source - 1 ? first + 1 : first;
        taps.second_weight[i] = second_weight;
        taps.first_weight[i] = 1.0f - second_weight;
    }
}

void compute_column_weight_lanes(const UpsampleTaps & taps, int width, float * lanes) {
    for (int x = 0; x < width; x += 4) {
        float * first = lanes + static_cast<size_t>(x / 4) * 64;
        for (int j = 0; j < 4; j++) {
            for (int c = 0; c < 8; c++) {
                first[8 * j + c] = taps.first_weight[x + j];
                first[32 + 8 * j + c] = taps.second_weight[x + j];
            }
        }
    }
}

runtime::Status run_score_head(runtime::Session & session, const ScoreHeadArguments & arguments) {
    ScoreJob * kept = session.state<ScoreJob>();
    if (kept == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    ScoreJob & job = *kept;

    for (int k = 1; k < 4; k++) {
        auto * lanes =
            reinterpret_cast<Vector<float> *>(arguments.vtcm + static_cast<size_t>(k - 1) * arguments.width * 64);
        const auto * source = hvx::vectors(arguments.column_weight_lanes[k]);
        for (int v = 0; v < arguments.width / 2; v++) lanes[v] = source[v];
        job.weight_lanes[k] = lanes;
    }

    job.arguments = &arguments;
    job.score1 = lane_selu_constants(arguments.score1_selu);
    float score1_zero_low[4];
    float score1_zero_high[4];
    for (int c = 0; c < 4; c++) {
        score1_zero_low[c] = arguments.score_zero[0][c];
        score1_zero_high[c] = arguments.score_zero[0][4 + c];
    }
    job.pads[0] = pad_words(score1_zero_low);
    job.pads[1] = pad_words(score1_zero_high);
    job.pads[2] = pad_words(arguments.score_zero[1]);
    job.pads[3] = pad_words(arguments.score_zero[2]);

    job.layers[0] = ScoreHeadTensor(arguments.layer_tensors[0], 4, 2);
    job.layers[1] = ScoreHeadTensor(arguments.layer_tensors[1], 4, 1);
    job.layers[2] = ScoreHeadTensor(arguments.layer_tensors[2], 1, 1);
    for (int index = 0; index < 2; index++) {
        for (int n = 0; n < 4; n++) {
            job.layer_constants[index][n] =
                splat_selu_constants(job.layers[index].constants + convolution_int8::kSeluConstantCount * n);
        }
    }
    job.bands = runtime::kHvxThreads;

    const runtime::Status status = runtime::run_parallel(
        session, runtime::kHvxThreads, [](void * job_pointer, int thread) { score_band(job_pointer, thread, thread); },
        &job);
    if (!status) return status;

    uint64_t upsampling = 0;
    for (int thread = 0; thread < runtime::kHvxThreads; thread++) upsampling += job.upsampling_microseconds[thread];
    arguments.statistics[stat::kScoreUpsamplingMicroseconds] = static_cast<int32_t>(upsampling / runtime::kHvxThreads);
    return {};
}

constexpr int kRadius = 2;
constexpr float kSoftArgmaxScale = 14.426950408889634f;  // 1 / temperature (0.1) * log2 e
constexpr int kItemKeypoints = 8;

// alignas(8): the workers load adjacent fields in pairs (memd).
struct alignas(8) DetectionArguments {
    const float * score_map;   // [height][width] (width a multiple of 32)
    const float * nms_scores;  // the same values (e.g. a copy in VTCM), read by the NMS and the candidate scan
    int height;
    int width;
    float threshold;
    int max_keypoints;
    float * maps[4];           // workspace: four [height][width] fp32 maps
    float * candidate_scores;  // workspace: height * width / 4 entries each
    int32_t * candidate_positions;
    float * sort_scores;  // workspace for the sort, the same sizes
    int32_t * sort_positions;
    // outputs
    float * keypoints;   // [k][2] pixel coordinates (x, y)
    float * normalized;  // [k][2] in [-1, 1]
    float * scores;      // [k]: the score map's bilinear samples at the keypoints
    int * keypoint_count;
    int32_t * statistics;
};

inline Vector<float> previous_lanes(Vector<float> current, Vector<float> previous, int lanes) {
    return slide_up(current, previous, 4 * lanes);
}
inline Vector<float> next_lanes(Vector<float> current, Vector<float> next, int lanes) {
    return slide_down(next, current, 4 * lanes);
}

// The max pools pad with zeros: max_pool2d's -inf padding, as the scores and masks are >= 0.
void horizontal_max_row(const float * input, float * output, int width) {
    const auto * row = hvx::vectors(input);
    auto * out = hvx::vectors(output);
    const int vectors = width / 32;
    const Vector<float> zero = Vector<float>::zero();

    for (int v = 0; v < vectors; v++) {
        const Vector<float> previous = v > 0 ? row[v - 1] : zero;
        const Vector<float> current = row[v];
        const Vector<float> next = v + 1 < vectors ? row[v + 1] : zero;
        Vector<float> m = max(current, previous_lanes(current, previous, 1));
        m = max(m, previous_lanes(current, previous, 2));
        m = max(m, next_lanes(current, next, 1));
        out[v] = max(m, next_lanes(current, next, 2));
    }
}

inline Vector<float> vertical_max(const float * map, int width, int height, int y, int v) {
    Vector<float> m = Vector<float>::zero();
    for (int dy = -kRadius; dy <= kRadius; dy++) {
        if (y + dy < 0 || y + dy >= height) continue;
        m = max(m, hvx::vectors(map + static_cast<size_t>(y + dy) * width)[v]);
    }
    return m;
}

constexpr int kHorizontalMax = 0;
constexpr int kInitialMask = 1;
constexpr int kSuppression = 2;
constexpr int kMaskUpdate = 3;

struct RowPass {
    const DetectionArguments * a;
    const float * horizontal_input;
    int kind;
};

void row_pass(void * context, int y, int) {
    const auto * pass = static_cast<const RowPass *>(context);
    const DetectionArguments & a = *pass->a;
    const int W = a.width;
    const size_t offset = static_cast<size_t>(y) * W;
    const auto * scores = hvx::vectors(a.nms_scores + offset);
    auto * mask = hvx::vectors(a.maps[1] + offset);
    auto * suppressed = hvx::vectors(a.maps[2] + offset);
    auto * supp = hvx::vectors(a.maps[3] + offset);
    const Vector<float> one = Vector<float>::splat(1.0f);
    const Vector<float> zero = Vector<float>::zero();

    switch (pass->kind) {
        case kHorizontalMax:
            horizontal_max_row(pass->horizontal_input + offset, a.maps[0] + offset, W);
            return;
        case kInitialMask:  // max_mask = S == maxpool(S)
            for (int v = 0; v < W / 32; v++) {
                const Vector<float> m = vertical_max(a.maps[0], W, a.height, y, v);
                mask[v] = select(reinterpret<int32_t>(scores[v]) == reinterpret<int32_t>(m), one, zero);
            }
            return;
        case kSuppression:  // supp = maxpool(max_mask) > 0, S' = supp ? 0 : S
            for (int v = 0; v < W / 32; v++) {
                const Vector<float> m = vertical_max(a.maps[0], W, a.height, y, v);
                const Mask s = m > zero;
                supp[v] = select(s, one, zero);
                suppressed[v] = select(s, zero, scores[v]);
            }
            return;
        default:  // kMaskUpdate: max_mask |= (S' == maxpool(S')) & !supp
            for (int v = 0; v < W / 32; v++) {
                const Vector<float> m = vertical_max(a.maps[0], W, a.height, y, v);
                const Mask is_new = reinterpret<int32_t>(suppressed[v]) == reinterpret<int32_t>(m);
                const Mask not_supp = reinterpret<int32_t>(supp[v]) == reinterpret<int32_t>(zero);
                mask[v] = select(is_new & not_supp, one, mask[v]);
            }
            return;
    }
}

runtime::Status run_row_pass(runtime::Session & session, RowPass * pass, int kind) {
    pass->kind = kind;
    return runtime::parallel_for(session, runtime::kHvxThreads, pass->a->height, row_pass, pass);
}

struct CandidateJob {
    const DetectionArguments * a;
    float threshold;
    int counts[4];
    int capacity;
};

int candidate_total(const CandidateJob & job) {
    return job.counts[0] + job.counts[1] + job.counts[2] + job.counts[3];
}

void collect_candidates(void * job_pointer, int band) {
    auto * job = static_cast<CandidateJob *>(job_pointer);
    const DetectionArguments & a = *job->a;
    const int H = a.height;
    const int W = a.width;
    const int first = H * band / 4;
    const int end = H * (band + 1) / 4;
    float * scores = a.candidate_scores + static_cast<size_t>(band) * job->capacity;
    int32_t * positions = a.candidate_positions + static_cast<size_t>(band) * job->capacity;
    const Vector<float> threshold = Vector<float>::splat(job->threshold);
    const Vector<uint32_t> lane_bits = hvx::load(hvx::kWordLaneBits.bits);
    const Vector<float> one = Vector<float>::splat(1.0f);
    int count = 0;
    alignas(128) uint32_t reduced[32];

    for (int y = first; y < end; y++) {
        if (y < kRadius || y >= H - kRadius) continue;

        const float * score_row = a.nms_scores + static_cast<size_t>(y) * W;
        const auto * scores_v = hvx::vectors(score_row);
        const auto * mask = hvx::vectors(a.maps[1] + static_cast<size_t>(y) * W);
        for (int v = 0; v < W / 32; v++) {
            const Mask candidate =
                (reinterpret<int32_t>(mask[v]) == reinterpret<int32_t>(one)) & (scores_v[v] > threshold);
            Vector<uint32_t> bits = select(candidate, lane_bits, Vector<uint32_t>::zero());
            bits = vector_math::or_to_all_lanes(bits);
            store(reduced, bits);
            for (uint32_t m = reduced[0]; m != 0; m &= m - 1) {
                const int x = 32 * v + static_cast<int>(Q6_R_ct0_R(m));
                if (x < kRadius || x >= W - kRadius || count >= job->capacity) continue;
                scores[count] = score_row[x];
                positions[count++] = (y << 16) | x;
            }
        }
    }
    job->counts[band] = count;
}

// A stable LSD radix sort on ~bits: for scores >= 0 it orders by descending score.
void sort_descending(float * scores, int32_t * positions, float * other_scores, int32_t * other_positions, int count,
                     float ** sorted_scores, int32_t ** sorted_positions) {
    int histograms[4][256] = {};
    for (int c = 0; c < count; c++) {
        const uint32_t key = ~std::bit_cast<uint32_t>(scores[c]);
        for (int digit = 0; digit < 4; digit++) histograms[digit][(key >> (8 * digit)) & 255]++;
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

    *sorted_scores = scores;
    *sorted_positions = positions;
}

struct RefinementJob {
    const DetectionArguments * a;
    const int32_t * positions;
    int count;
};

// grid_sample (align_corners True) of the score map at normalized (nx, ny).
float bilinear_score(const DetectionArguments & a, float nx, float ny) {
    const float width_scale = static_cast<float>(a.width - 1);
    const float height_scale = static_cast<float>(a.height - 1);

    const float x_shifted = nx + 1.0f;
    const float x_halved = x_shifted / 2.0f;
    const float ix = x_halved * width_scale;
    const float y_shifted = ny + 1.0f;
    const float y_halved = y_shifted / 2.0f;
    const float iy = y_halved * height_scale;

    const float ix_nw = floorf(ix);
    const float iy_nw = floorf(iy);
    const float ix_se = ix_nw + 1.0f;
    const float iy_se = iy_nw + 1.0f;

    const float east = ix - ix_nw;
    const float west = ix_se - ix;
    const float south = iy - iy_nw;
    const float north = iy_se - iy;
    const float weights[4] = {west * north, east * north, west * south, east * south};
    const int xs[4] = {static_cast<int>(ix_nw), static_cast<int>(ix_se), static_cast<int>(ix_nw),
                       static_cast<int>(ix_se)};
    const int ys[4] = {static_cast<int>(iy_nw), static_cast<int>(iy_nw), static_cast<int>(iy_se),
                       static_cast<int>(iy_se)};

    float value = 0.0f;
    for (int c = 0; c < 4; c++) {
        if (xs[c] < 0 || xs[c] >= a.width || ys[c] < 0 || ys[c] >= a.height) continue;
        const float term = a.score_map[static_cast<size_t>(ys[c]) * a.width + xs[c]] * weights[c];
        value = value + term;
    }
    return value;
}

// One fp32 operation per statement here and in bilinear_score: no FMA contraction, as in the reference.
void refine_keypoints(void * context, int item, int) {
    const auto * job = static_cast<const RefinementJob *>(context);
    const DetectionArguments & a = *job->a;
    const float width_scale = static_cast<float>(a.width - 1);
    const float height_scale = static_cast<float>(a.height - 1);
    const int first = kItemKeypoints * item;
    alignas(128) float window[32];

    for (int k = first; k < first + kItemKeypoints && k < job->count; k++) {
        const int x = job->positions[k] & 0xffff;
        const int y = job->positions[k] >> 16;
        for (int dy = 0; dy < 5; dy++) {
            for (int dx = 0; dx < 5; dx++) {
                window[5 * dy + dx] = a.score_map[static_cast<size_t>(y - kRadius + dy) * a.width + x - kRadius + dx];
            }
        }

        float maximum = window[0];
        for (int t = 1; t < 25; t++) maximum = window[t] > maximum ? window[t] : maximum;
        for (int t = 25; t < 32; t++) window[t] = maximum;
        const Vector<float> exponents = (hvx::load(window) - maximum) * kSoftArgmaxScale;
        store(window, vector_math::exp2(exponents));

        float total = 0.0f;
        float sum_x = 0.0f;
        float sum_y = 0.0f;
        for (int t = 0; t < 25; t++) {
            const float e = window[t];
            const float weighted_x = e * static_cast<float>(t % 5 - kRadius);
            const float weighted_y = e * static_cast<float>(t / 5 - kRadius);
            total = total + e;
            sum_x = sum_x + weighted_x;
            sum_y = sum_y + weighted_y;
        }

        const float residual_x = sum_x / total;
        const float residual_y = sum_y / total;
        const float kx = static_cast<float>(x) + residual_x;
        const float ky = static_cast<float>(y) + residual_y;
        const float ratio_x = kx / width_scale;
        const float doubled_x = ratio_x * 2.0f;
        const float nx = doubled_x - 1.0f;
        const float ratio_y = ky / height_scale;
        const float doubled_y = ratio_y * 2.0f;
        const float ny = doubled_y - 1.0f;
        a.normalized[2 * k] = nx;
        a.normalized[2 * k + 1] = ny;
        a.scores[k] = bilinear_score(a, nx, ny);

        // the pixel coordinates from the normalized ones, as the reference (the round trip rounds)
        const float shifted_x = nx + 1.0f;
        const float scaled_x = width_scale * shifted_x;
        a.keypoints[2 * k] = scaled_x / 2.0f;
        const float shifted_y = ny + 1.0f;
        const float scaled_y = height_scale * shifted_y;
        a.keypoints[2 * k + 1] = scaled_y / 2.0f;
    }
}

runtime::Status run_detection(runtime::Session & session, const DetectionArguments & arguments) {
    runtime::StageTimer nms_timer(arguments.statistics);
    RowPass pass{&arguments, arguments.nms_scores, kHorizontalMax};
    runtime::Status status = run_row_pass(session, &pass, kHorizontalMax);
    if (!status) return status;
    status = run_row_pass(session, &pass, kInitialMask);
    if (!status) return status;

    for (int round = 0; round < 2; round++) {
        pass.horizontal_input = arguments.maps[1];
        status = run_row_pass(session, &pass, kHorizontalMax);
        if (!status) return status;
        status = run_row_pass(session, &pass, kSuppression);
        if (!status) return status;
        pass.horizontal_input = arguments.maps[2];
        status = run_row_pass(session, &pass, kHorizontalMax);
        if (!status) return status;
        status = run_row_pass(session, &pass, kMaskUpdate);
        if (!status) return status;
    }
    nms_timer.lap(stat::kNmsMicroseconds);

    const int pixels = arguments.height * arguments.width;
    CandidateJob candidates{&arguments, arguments.threshold, {}, pixels / 16};
    status = runtime::run_parallel(session, runtime::kHvxThreads, collect_candidates, &candidates);
    if (!status) return status;
    int total = candidate_total(candidates);
    if (total == 0) {
        double sum = 0.0;
        for (int i = 0; i < pixels; i++) sum += static_cast<double>(arguments.score_map[i]);
        candidates.threshold = static_cast<float>(sum / static_cast<double>(pixels));
        status = runtime::run_parallel(session, runtime::kHvxThreads, collect_candidates, &candidates);
        if (!status) return status;
        total = candidate_total(candidates);
    }
    arguments.statistics[stat::kCandidates] = total;

    float * scores = arguments.sort_scores;
    int32_t * positions = arguments.sort_positions;
    int count = 0;
    for (int band = 0; band < runtime::kHvxThreads; band++) {
        memmove(scores + count, arguments.candidate_scores + static_cast<size_t>(band) * candidates.capacity,
                static_cast<size_t>(candidates.counts[band]) * 4);
        memmove(positions + count, arguments.candidate_positions + static_cast<size_t>(band) * candidates.capacity,
                static_cast<size_t>(candidates.counts[band]) * 4);
        count += candidates.counts[band];
    }
    float * sorted_scores;
    int32_t * sorted_positions;
    sort_descending(scores, positions, arguments.candidate_scores, arguments.candidate_positions, count, &sorted_scores,
                    &sorted_positions);

    const int keypoints = count < arguments.max_keypoints ? count : arguments.max_keypoints;
    RefinementJob refinement{&arguments, sorted_positions, keypoints};
    status = runtime::parallel_for(session, runtime::kHvxThreads, (keypoints + kItemKeypoints - 1) / kItemKeypoints,
                                   refine_keypoints, &refinement);
    if (!status) return status;
    *arguments.keypoint_count = keypoints;
    return {};
}

constexpr int kSamples = 16;
constexpr int kPatchBytes = 9 * 128;
constexpr int kSortRowsStride = kMaximumDescriptorHeight + 2;
constexpr int kItemPatches = 4;
constexpr int kItemSamples = 8;       // x 4 corners <= 32: one normalize_batch
constexpr int kItemDescriptors = 32;  // <= 32: one normalize_batch
constexpr int kPrefetchItems = 8;

// - height: at most kMaximumDescriptorHeight.
// - aggregations[k]: agg_{k+1}'s codes [height / f][width / f][32], f = 1, 2, 8, 32.
// - column_taps / row_taps: [4], the upsampling taps of parts 1 .. 3 ([0] unused).
// - workspace for up to `capacity` keypoints (a multiple of 64), each with room for 8 more rows: patches
//   [capacity][1152], hidden [capacity][32], offsets [capacity][32] fp32, samples / features [16 capacity][128],
//   descriptors_f32 [capacity][128].
// alignas(8): the workers load adjacent fields in pairs (memd).
struct alignas(8) DescriptorArguments {
    int32_t * statistics;
    int height;
    int width;
    const uint8_t * aggregations[4];
    const UpsampleTaps * column_taps;
    const UpsampleTaps * row_taps;
    const StageConstants * constants;
    const uint8_t * offset0_weights;
    const uint8_t * offset0_aux;
    const uint8_t * offset2_weights;
    const uint8_t * features_weights;
    const uint8_t * features_aux;
    const uint8_t * aggregation_weights;
    const float * keypoints;  // normalized [count][2]
    int count;
    int capacity;
    uint8_t * patches;
    uint8_t * hidden;
    float * offsets;
    uint8_t * samples;
    uint8_t * features;
    float * descriptors_f32;
    _Float16 * descriptors;        // output: [count][128], normalized
    hvx::Vector<float> * scratch;  // per thread kDescriptorScratchVectors
    float * small_maps;            // workspace: [(h / 8)(w / 8) + (h / 32)(w / 32)][32] fp32
    SamplePosition * positions;    // workspace: [16 capacity]
    int * sample_order;            // workspace: [16 capacity]
    int * row_counts;              // workspace: [kSortSlices][kMaximumDescriptorHeight + 2]
};

inline Vector<int32_t> codes_as_words(const uint8_t * codes) {
    const VectorPair<uint16_t> halfwords = widen(hvx::load_unaligned(codes));
    return reinterpret<int32_t>(widen(halfwords.low()).low());
}

inline Vector<float> dequantize(const uint8_t * codes, Vector<float> zero, Vector<float> scale) {
    return (to_float(codes_as_words(codes)) - zero) * scale;
}

struct CodeMap {
    const uint8_t * codes;
    int width;
    Vector<float> zero;
    Vector<float> scale;
    Vector<float> at(int row, int column) const {
        return dequantize(codes + (static_cast<size_t>(row) * width + column) * 32, zero, scale);
    }
};
struct ValueMap {
    const float * values;
    int width;
    Vector<float> at(int row, int column) const {
        return hvx::load(values + (static_cast<size_t>(row) * width + column) * 32);
    }
};

struct FeatureMaps {
    int height;
    int width;
    CodeMap agg1;
    CodeMap agg2;
    ValueMap agg3;
    ValueMap agg4;
    UpsampleTaps row_taps[4];
    UpsampleTaps column_taps[4];
    bool agg2_in_vtcm;
};

struct FeatureContext {
    const DescriptorArguments * a;
    FeatureMaps maps;
};

inline int clamp_index(int v, int size) {
    return v < 0 ? 0 : v >= size ? size - 1 : v;
}

template <int Rows, int Columns, class Map>
inline void upsampled_block(const Map & map, const UpsampleTaps & row_taps, const UpsampleTaps & column_taps,
                            const int * rows, const int * columns, Vector<float> * __restrict out) {
    Vector<float> horizontal[Rows][2][Columns];
#pragma clang loop unroll(full)
    for (int i = 0; i < Rows; i++) {
        const int source_rows[2] = {row_taps.first[rows[i]], row_taps.second[rows[i]]};
#pragma clang loop unroll(full)
        for (int k = 0; k < 2; k++) {
#pragma clang loop unroll(full)
            for (int j = 0; j < Columns; j++) {
                const int c = columns[j];
                const Vector<float> left = map.at(source_rows[k], column_taps.first[c]);
                const Vector<float> right = map.at(source_rows[k], column_taps.second[c]);
                horizontal[i][k][j] = column_taps.first_weight[c] * left + column_taps.second_weight[c] * right;
            }
        }
    }

#pragma clang loop unroll(full)
    for (int i = 0; i < Rows; i++) {
        const Vector<float> m0 = Vector<float>::splat(row_taps.first_weight[rows[i]]);
        const Vector<float> m1 = Vector<float>::splat(row_taps.second_weight[rows[i]]);
#pragma clang loop unroll(full)
        for (int j = 0; j < Columns; j++) {
            out[4 * (Columns * i + j)] = m0 * horizontal[i][0][j] + m1 * horizontal[i][1][j];
        }
    }
}

// The feature map's unnormalized values: out[4 (Columns i + j) + k] holds channels 32 k .. 32 k + 31 of pixel
// (y + i, x + j), clamped into the map.
template <int Rows, int Columns>
inline void feature_block(const FeatureMaps & maps, int y, int x, Vector<float> * __restrict out) {
    int rows[Rows];
    int columns[Columns];
#pragma clang loop unroll(full)
    for (int i = 0; i < Rows; i++) rows[i] = clamp_index(y + i, maps.height);
#pragma clang loop unroll(full)
    for (int j = 0; j < Columns; j++) columns[j] = clamp_index(x + j, maps.width);

#pragma clang loop unroll(full)
    for (int i = 0; i < Rows; i++) {
#pragma clang loop unroll(full)
        for (int j = 0; j < Columns; j++) out[4 * (Columns * i + j)] = maps.agg1.at(rows[i], columns[j]);
    }
    upsampled_block<Rows, Columns>(maps.agg2, maps.row_taps[1], maps.column_taps[1], rows, columns, out + 1);
    upsampled_block<Rows, Columns>(maps.agg3, maps.row_taps[2], maps.column_taps[2], rows, columns, out + 2);
    upsampled_block<Rows, Columns>(maps.agg4, maps.row_taps[3], maps.column_taps[3], rows, columns, out + 3);
}

// The 2 x 2 corners from 3 x 3 sources: adjacent pixels' taps are equal or chained (compute_upsample_taps), so pixel
// row 1 takes source rows 0, 1 or 1, 2 (columns likewise).
template <class Map>
inline void upsampled_corners(const Map & map, const UpsampleTaps & row_taps, const UpsampleTaps & column_taps,
                              const int rows[2], const int columns[2], Vector<float> * __restrict out) {
    const int source_rows[3] = {row_taps.first[rows[0]], row_taps.second[rows[0]], row_taps.second[rows[1]]};
    const int source_columns[3] = {column_taps.first[columns[0]], column_taps.second[columns[0]],
                                   column_taps.second[columns[1]]};
    const Mask rows_chained = Mask::uniform(row_taps.first[rows[1]] != source_rows[0]);
    const Mask columns_chained = Mask::uniform(column_taps.first[columns[1]] != source_columns[0]);

    Vector<float> sources[3][3];
#pragma clang loop unroll(full)
    for (int r = 0; r < 3; r++) {
#pragma clang loop unroll(full)
        for (int c = 0; c < 3; c++) sources[r][c] = map.at(source_rows[r], source_columns[c]);
    }

    const Vector<float> l00 = Vector<float>::splat(column_taps.first_weight[columns[0]]);
    const Vector<float> l01 = Vector<float>::splat(column_taps.second_weight[columns[0]]);
    const Vector<float> l10 = Vector<float>::splat(column_taps.first_weight[columns[1]]);
    const Vector<float> l11 = Vector<float>::splat(column_taps.second_weight[columns[1]]);
    Vector<float> horizontal[3][2];
#pragma clang loop unroll(full)
    for (int r = 0; r < 3; r++) {
        horizontal[r][0] = l00 * sources[r][0] + l01 * sources[r][1];
        const Vector<float> left = select(columns_chained, sources[r][1], sources[r][0]);
        const Vector<float> right = select(columns_chained, sources[r][2], sources[r][1]);
        horizontal[r][1] = l10 * left + l11 * right;
    }

    const Vector<float> m00 = Vector<float>::splat(row_taps.first_weight[rows[0]]);
    const Vector<float> m01 = Vector<float>::splat(row_taps.second_weight[rows[0]]);
    const Vector<float> m10 = Vector<float>::splat(row_taps.first_weight[rows[1]]);
    const Vector<float> m11 = Vector<float>::splat(row_taps.second_weight[rows[1]]);
#pragma clang loop unroll(full)
    for (int j = 0; j < 2; j++) {
        out[4 * j] = m00 * horizontal[0][j] + m01 * horizontal[1][j];
        const Vector<float> top = select(rows_chained, horizontal[1][j], horizontal[0][j]);
        const Vector<float> bottom = select(rows_chained, horizontal[2][j], horizontal[1][j]);
        out[4 * (2 + j)] = m10 * top + m11 * bottom;
    }
}

// feature_block<2, 2>'s values, from shared sources.
inline void corner_feature_vectors(const FeatureMaps & maps, int y, int x, Vector<float> * __restrict out) {
    const int rows[2] = {clamp_index(y, maps.height), clamp_index(y + 1, maps.height)};
    const int columns[2] = {clamp_index(x, maps.width), clamp_index(x + 1, maps.width)};
    Vector<float> agg1[4];
#pragma clang loop unroll(full)
    for (int corner = 0; corner < 4; corner++) agg1[corner] = maps.agg1.at(rows[corner / 2], columns[corner % 2]);
    upsampled_corners(maps.agg2, maps.row_taps[1], maps.column_taps[1], rows, columns, out + 1);
    upsampled_corners(maps.agg3, maps.row_taps[2], maps.column_taps[2], rows, columns, out + 2);
    upsampled_corners(maps.agg4, maps.row_taps[3], maps.column_taps[3], rows, columns, out + 3);
#pragma clang loop unroll(full)
    for (int corner = 0; corner < 4; corner++) out[4 * corner] = agg1[corner];
}

// As device_model.lane_sum_of_squares: lane-wise in channel order, then the rotate-and-add tree; lane 0 is the sum.
inline Vector<float> squares_tree(const Vector<float> v[4]) {
    Vector<float> lanes = v[0] * v[0] + v[1] * v[1];
    lanes = lanes + v[2] * v[2];
    lanes = lanes + v[3] * v[3];
    return vector_math::sum_to_all_lanes(lanes);
}

// Normalizes `count` (<= 32) vectors of 128 lanes (vectors[4 i .. 4 i + 3]) in place, as device_model.normalized.
void normalize_batch(Vector<float> * vectors, int count) {
    Vector<float> sums = Vector<float>::zero();
    for (int i = 0; i < count; i++) {
        const Vector<float> lanes = squares_tree(vectors + 4 * i);
        sums = slide_down(lanes, sums, 4);
    }
    if (count < 32) sums = slide_down(Vector<float>::zero(), sums, 4 * (32 - count));  // lane i: vector i

    const Vector<float> reciprocal = min(vector_math::precise_reciprocal_square_root(sums), 1e12f);
    alignas(128) float scales[32];
    store(scales, reciprocal);

    for (int i = 0; i < count; i++) {
        const Vector<float> scale = Vector<float>::splat(scales[i]);
        for (int k = 0; k < 4; k++) vectors[4 * i + k] = vectors[4 * i + k] * scale;
    }
}

inline Vector<uint8_t> quantize(const Vector<float> v[4], const float * r, const float * z) {
    Vector<int32_t> codes[4];
    for (int k = 0; k < 4; k++) {
        const Vector<float> rk = hvx::load_unaligned(r + 32 * k);
        const Vector<float> zk = hvx::load_unaligned(z + 32 * k);
        codes[k] = truncate_to_int32(v[k] * rk + zk);
    }
    return pack_codes(codes[0], codes[1], codes[2], codes[3]);
}

struct KeypointGeometry {
    float x;
    float y;
};

inline KeypointGeometry keypoint_pixels(const float * keypoints, int n, float width_scale, float height_scale) {
    const float half_x = keypoints[2 * n] / 2.0f;
    const float shifted_x = half_x + 0.5f;
    const float half_y = keypoints[2 * n + 1] / 2.0f;
    const float shifted_y = half_y + 0.5f;
    return {shifted_x * width_scale, shifted_y * height_scale};
}

inline Vector<float> * thread_scratch(const DescriptorArguments & a, int thread) {
    return a.scratch + static_cast<size_t>(thread) * kDescriptorScratchVectors;
}

CodeMap code_map(const DescriptorArguments & a, int part) {
    constexpr int kDownsampling[4] = {1, 2, 8, 32};
    return {a.aggregations[part], a.width / kDownsampling[part],
            hvx::load_unaligned(a.constants->aggregation_zero[part]),
            hvx::load_unaligned(a.constants->aggregation_scale[part])};
}

void dequantize_small_maps(void * context, int item, int) {
    const auto * f = static_cast<const FeatureContext *>(context);
    const DescriptorArguments & a = *f->a;
    const int agg3_rows = a.height / 8;
    const bool agg3 = item < agg3_rows;
    const CodeMap codes = code_map(a, agg3 ? 2 : 3);
    const ValueMap & values = agg3 ? f->maps.agg3 : f->maps.agg4;
    const int row = agg3 ? item : item - agg3_rows;
    auto * out = hvx::vectors(const_cast<float *>(values.values) + static_cast<size_t>(row) * values.width * 32);
    for (int x = 0; x < values.width; x++) out[x] = codes.at(row, x);
}

void patch_codes(void * context, int item, int thread) {
    const auto * f = static_cast<const FeatureContext *>(context);
    const DescriptorArguments & a = *f->a;
    const FeatureMaps maps = f->maps;
    const int first = kItemPatches * item;
    Vector<float> * window = thread_scratch(a, thread);

    for (int n = first; n < first + kItemPatches && n < a.count; n++) {
        const KeypointGeometry k =
            keypoint_pixels(a.keypoints, n, static_cast<float>(a.width - 1), static_cast<float>(a.height - 1));
        // LightGlue's get_patches (patch size 3)
        int corner_x = static_cast<int>(static_cast<float>(static_cast<int>(k.x)) - 0.5f);
        int corner_y = static_cast<int>(static_cast<float>(static_cast<int>(k.y)) - 0.5f);
        corner_x = corner_x < 0 ? 0 : corner_x > a.width - 4 ? a.width - 4 : corner_x;
        corner_y = corner_y < 0 ? 0 : corner_y > a.height - 4 ? a.height - 4 : corner_y;

        for (int row = 0; row < 3; row++) feature_block<1, 3>(maps, corner_y + row, corner_x, window + 12 * row);
        normalize_batch(window, 9);

        auto * row = hvx::vectors(a.patches + static_cast<size_t>(n) * kPatchBytes);
        for (int tap = 0; tap < 9; tap++)
            row[tap] = quantize(window + 4 * tap, a.constants->patch_r, a.constants->patch_z);
    }
}

inline float clamp_offset(float v, float limit) {
    const float low = v < -limit ? -limit : v;
    return low > limit ? limit : low;
}
inline int floor_to_int(float v) {
    const int truncated = static_cast<int>(v);
    return v < static_cast<float>(truncated) ? truncated - 1 : truncated;
}

inline int sort_row(int y, int height) {
    const int clamped = y < -1 ? -1 : y > height - 1 ? height - 1 : y;
    return clamped + 1;
}

// As SDDH and grid_sample: the positions normalized and unnormalized again (the round trip rounds), one fp32
// operation per statement (no FMA contraction).
void plan_positions(void * context, int slice, int) {
    const auto * f = static_cast<const FeatureContext *>(context);
    const DescriptorArguments & a = *f->a;
    const int height = a.height;
    const float limit = static_cast<float>(height > a.width ? height : a.width) / 4.0f;
    const float width_scale = static_cast<float>(a.width - 1);
    const float height_scale = static_cast<float>(height - 1);
    int * counts = a.row_counts + static_cast<size_t>(slice) * kSortRowsStride;
    memset(counts, 0, static_cast<size_t>(height + 1) * 4);
    const int end = a.count * (slice + 1) / kSortSlices;

    for (int n = a.count * slice / kSortSlices; n < end; n++) {
        const KeypointGeometry k = keypoint_pixels(a.keypoints, n, width_scale, height_scale);
        const float * offsets = a.offsets + static_cast<size_t>(n) * 32;
        for (int p = 0; p < kSamples; p++) {
            const float dx = clamp_offset(offsets[p], limit);
            const float dy = clamp_offset(offsets[kSamples + p], limit);
            const float position_x = k.x + dx;
            const float position_y = k.y + dy;

            const float doubled_x = 2.0f * position_x;
            const float ratio_x = doubled_x / width_scale;
            const float nx = ratio_x - 1.0f;
            const float doubled_y = 2.0f * position_y;
            const float ratio_y = doubled_y / height_scale;
            const float ny = ratio_y - 1.0f;

            const float shifted_x = nx + 1.0f;
            const float halved_x = shifted_x * 0.5f;  // exact: / 2
            const float ix = halved_x * width_scale;
            const float shifted_y = ny + 1.0f;
            const float halved_y = shifted_y * 0.5f;
            const float iy = halved_y * height_scale;

            const int x0 = floor_to_int(ix);
            const int y0 = floor_to_int(iy);
            const float ix_nw = static_cast<float>(x0);
            const float iy_nw = static_cast<float>(y0);
            const float ix_se = ix_nw + 1.0f;
            const float iy_se = iy_nw + 1.0f;

            const float east = ix - ix_nw;
            const float west = ix_se - ix;
            const float south = iy - iy_nw;
            const float north = iy_se - iy;

            SamplePosition & sample = a.positions[kSamples * n + p];
            sample.y = y0;
            sample.x = x0;
            sample.weight[0] = west * north;
            sample.weight[1] = east * north;
            sample.weight[2] = west * south;
            sample.weight[3] = east * south;
            counts[sort_row(y0, height)]++;
        }
    }
}

void scatter_order(void * context, int slice, int) {
    const auto * f = static_cast<const FeatureContext *>(context);
    const DescriptorArguments & a = *f->a;
    const int height = a.height;
    int * next = a.row_counts + static_cast<size_t>(slice) * kSortRowsStride;
    const int end = kSamples * (a.count * (slice + 1) / kSortSlices);
    for (int i = kSamples * (a.count * slice / kSortSlices); i < end; i++) {
        a.sample_order[next[sort_row(a.positions[i].y, height)]++] = i;
    }
}

inline int item_bottom_row(const DescriptorArguments & a, int item) {
    const int total = kSamples * a.count;
    const int last = (kItemSamples * (item + 1) < total ? kItemSamples * (item + 1) : total) - 1;
    return clamp_index(a.positions[a.sample_order[last]].y + 1, a.height);
}

inline void prefetch_rows_after(const uint8_t * codes, int width, int previous, int bottom) {
    const int first = previous + 1 > bottom - 3 ? previous + 1 : bottom - 3;
    if (first > bottom) return;
    const uint32_t row_bytes = static_cast<uint32_t>(width) * 32;
    hvx::prefetch_to_l2(codes + first * row_bytes, row_bytes / 4, row_bytes / 4, 4 * (bottom - first + 1));
}

void prefetch_new_rows(const DescriptorArguments & a, const FeatureMaps & maps, int item) {
    const int ahead = item + kPrefetchItems;
    if (kItemSamples * ahead >= kSamples * a.count) return;

    const int step = maps.agg2_in_vtcm ? 1 : 2;
    const int bottom = item_bottom_row(a, ahead);
    const int previous_bottom = ahead >= step ? item_bottom_row(a, ahead - step) : -1;
    if (maps.agg2_in_vtcm || item % 2 == 0) {
        prefetch_rows_after(maps.agg1.codes, maps.width, previous_bottom, bottom);
    } else {
        const UpsampleTaps & row_taps = maps.row_taps[1];
        const int source_bottom = row_taps.second[bottom];
        const int previous_source_bottom = previous_bottom >= 0 ? row_taps.second[previous_bottom] : -1;
        prefetch_rows_after(maps.agg2.codes, maps.agg2.width, previous_source_bottom, source_bottom);
    }
}

void sample_codes(void * context, int item, int thread) {
    const auto * f = static_cast<const FeatureContext *>(context);
    const DescriptorArguments & a = *f->a;
    const FeatureMaps maps = f->maps;
    const int total = kSamples * a.count;
    const int first = kItemSamples * item;
    const int count = total - first < kItemSamples ? total - first : kItemSamples;
    Vector<float> * corners = thread_scratch(a, thread);

    prefetch_new_rows(a, maps, item);
    for (int i = 0; i < count; i++) {
        const SamplePosition & sample = a.positions[a.sample_order[first + i]];
        corner_feature_vectors(maps, sample.y, sample.x, corners + 16 * i);
    }
    normalize_batch(corners, 4 * count);

    for (int i = 0; i < count; i++) {
        const int index = a.sample_order[first + i];
        const SamplePosition & sample = a.positions[index];
        Vector<float> value[4] = {Vector<float>::zero(), Vector<float>::zero(), Vector<float>::zero(),
                                  Vector<float>::zero()};
#pragma clang loop unroll(full)
        for (int c = 0; c < 4; c++) {
            const int cy = sample.y + c / 2;
            const int cx = sample.x + c % 2;
            const bool inside = cy >= 0 && cy < a.height && cx >= 0 && cx < a.width;
            const Vector<float> weight = Vector<float>::splat(inside ? sample.weight[c] : 0.0f);
            const Vector<float> * corner = corners + 16 * i + 4 * c;
#pragma clang loop unroll(full)
            for (int lane_group = 0; lane_group < 4; lane_group++) {
                value[lane_group] = value[lane_group] + corner[lane_group] * weight;
            }
        }

        hvx::vector_at<uint8_t>(a.samples + static_cast<size_t>(index) * 128) =
            quantize(value, a.constants->samples_r, a.constants->samples_z);
    }
}

void normalize_descriptors(void * context, int item, int thread) {
    const auto * f = static_cast<const FeatureContext *>(context);
    const DescriptorArguments & a = *f->a;
    const int first = kItemDescriptors * item;
    const int count = a.count - first < kItemDescriptors ? a.count - first : kItemDescriptors;
    Vector<float> * v = thread_scratch(a, thread);

    for (int i = 0; i < count; i++) {
        const auto * row = hvx::vectors(a.descriptors_f32 + static_cast<size_t>(first + i) * 128);
        for (int k = 0; k < 4; k++) v[4 * i + k] = row[k];
    }
    normalize_batch(v, count);

    for (int i = 0; i < count; i++) {
        _Float16 * out = a.descriptors + static_cast<size_t>(first + i) * 128;
        for (int half = 0; half < 2; half++) {  // two fp32 vectors -> one fp16 vector in lane order
            const VectorPair<float> dealt = deinterleave(v[4 * i + 2 * half + 1], v[4 * i + 2 * half]);
            store_unaligned(out + 64 * half, to_float16_interleaved(dealt.low(), dealt.high()));
        }
    }
}

runtime::Status descriptor_convolution(runtime::Session & session, const uint8_t * input, int rows, int row_width,
                                       int channels, const uint8_t * weights, int N, const uint8_t * aux,
                                       uint8_t * codes, float * rows_f32) {
    convolution_int8::Arguments c;
    c.input = input;
    c.height = rows / row_width;
    c.width = row_width;
    c.channels = channels;
    c.taps = 1;
    c.weights = weights;
    c.N = N;

    if (codes != nullptr) {
        c.uint8_outputs[0] = codes;
        c.uint8_output_channels = N;
        c.requantization = convolution_int8::Requantization::kSelu;
        c.selu_constants = reinterpret_cast<const float *>(aux);
    } else {
        c.output_form = convolution_int8::OutputForm::kRowsF32;
        c.rows_f32 = rows_f32;
    }
    return kernels::convolution_int8::run(session, c);
}

runtime::Status run_descriptors(runtime::Session & session, const DescriptorArguments & arguments) {
    const size_t agg3_pixels = static_cast<size_t>(arguments.height / 8) * (arguments.width / 8);
    alignas(128) FeatureContext f;
    f.a = &arguments;
    FeatureMaps & maps = f.maps;
    maps.height = arguments.height;
    maps.width = arguments.width;
    maps.agg1 = code_map(arguments, 0);
    maps.agg2 = code_map(arguments, 1);
    maps.agg3 = {arguments.small_maps, arguments.width / 8};
    maps.agg4 = {arguments.small_maps + agg3_pixels * 32, arguments.width / 32};
    for (int part = 1; part < 4; part++) {
        maps.row_taps[part] = arguments.row_taps[part];
        maps.column_taps[part] = arguments.column_taps[part];
    }
    maps.agg2_in_vtcm = false;

    runtime::StageTimer timer(arguments.statistics);
    runtime::Status status = runtime::parallel_for(
        session, runtime::kHvxThreads, arguments.height / 8 + arguments.height / 32, dequantize_small_maps, &f);
    if (!status) return status;
    const int rows = (arguments.count + 63) / 64 * 64;
    memset(arguments.patches + static_cast<size_t>(arguments.count) * kPatchBytes, 0,
           static_cast<size_t>(rows - arguments.count) * kPatchBytes);
    status = runtime::parallel_for(session, runtime::kHvxThreads, (arguments.count + kItemPatches - 1) / kItemPatches,
                                   patch_codes, &f);
    if (!status) return status;
    timer.lap(stat::kDescriptorPatchesMicroseconds);

    status = descriptor_convolution(session, arguments.patches, rows, 32, kPatchBytes, arguments.offset0_weights, 32,
                                    arguments.offset0_aux, arguments.hidden, nullptr);
    if (!status) return status;
    status = descriptor_convolution(session, arguments.hidden, rows, 32, 32, arguments.offset2_weights, 32, nullptr,
                                    nullptr, arguments.offsets);
    if (!status) return status;
    timer.lap(stat::kDescriptorConvolutionsMicroseconds);

    memset(arguments.samples + static_cast<size_t>(arguments.count) * kSamples * 128, 0,
           static_cast<size_t>(rows - arguments.count) * kSamples * 128);

    const size_t agg2_bytes = static_cast<size_t>(arguments.height / 2) * (arguments.width / 2) * 32;
    const size_t small_bytes =
        (agg3_pixels + static_cast<size_t>(arguments.height / 32) * (arguments.width / 32)) * 128;
    if (agg2_bytes + small_bytes <= session.vtcm_bytes()) {
        uint8_t * vtcm = session.vtcm();
        status = runtime::copy_memory_parallel(session, vtcm, maps.agg2.codes, agg2_bytes);
        if (!status) return status;
        status = runtime::copy_memory_parallel(session, vtcm + agg2_bytes, arguments.small_maps, small_bytes);
        if (!status) return status;
        maps.agg2.codes = vtcm;
        maps.agg3.values = reinterpret_cast<const float *>(vtcm + agg2_bytes);
        maps.agg4.values = maps.agg3.values + (maps.agg4.values - arguments.small_maps);
        maps.agg2_in_vtcm = true;
    }

    status = runtime::parallel_for(session, runtime::kHvxThreads, kSortSlices, plan_positions, &f);
    if (!status) return status;
    int place = 0;
    for (int row = 0; row <= arguments.height; row++) {
        for (int slice = 0; slice < kSortSlices; slice++) {
            int & count = arguments.row_counts[static_cast<size_t>(slice) * kSortRowsStride + row];
            const int samples = count;
            count = place;
            place += samples;
        }
    }
    status = runtime::parallel_for(session, runtime::kHvxThreads, kSortSlices, scatter_order, &f);
    if (!status) return status;

    status = runtime::parallel_for(session, runtime::kHvxThreads,
                                   (kSamples * arguments.count + kItemSamples - 1) / kItemSamples, sample_codes, &f);
    if (!status) return status;
    timer.lap(stat::kDescriptorSamplesMicroseconds);

    status = descriptor_convolution(session, arguments.samples, rows * kSamples, 64, 128, arguments.features_weights,
                                    128, arguments.features_aux, arguments.features, nullptr);
    if (!status) return status;
    status = descriptor_convolution(session, arguments.features, rows, 32, kSamples * 128,
                                    arguments.aggregation_weights, 128, nullptr, nullptr, arguments.descriptors_f32);
    if (!status) return status;
    timer.lap(stat::kDescriptorConvolutionsMicroseconds);

    return runtime::parallel_for(session, runtime::kHvxThreads,
                                 (arguments.count + kItemDescriptors - 1) / kItemDescriptors, normalize_descriptors,
                                 &f);
}

struct Workspace {
    int height;
    int width;
    // padded as Block1Conv1Arguments::image; the padding is zeroed once, at the allocation
    runtime::HeapArray<uint8_t> image;
    size_t image_row_bytes;
    runtime::HeapArray<uint8_t> block1_mid;
    runtime::HeapArray<uint8_t> x1;
    runtime::HeapArray<uint8_t> x1_pool;
    runtime::HeapArray<uint8_t> block2_concatenated;  // [block2.mid | x1.pool]
    runtime::HeapArray<uint8_t> x2;
    runtime::HeapArray<uint8_t> x2_pool;
    runtime::HeapArray<uint8_t> block3_mid;
    runtime::HeapArray<uint8_t> x3;
    runtime::HeapArray<uint8_t> x3_pool;
    runtime::HeapArray<uint8_t> block4_mid;
    runtime::HeapArray<uint8_t> x4;
    runtime::HeapArray<uint8_t> aggregations[4];
    runtime::HeapArray<float> parts[4];
    runtime::HeapArray<float> offsets;
    runtime::HeapArray<float> dequantized;
    runtime::HeapArray<uint8_t> columns;
    runtime::HeapArray<int> taps_ints;
    runtime::HeapArray<float> taps_floats;
    UpsampleTaps column_taps[4];
    UpsampleTaps row_taps[4];
    runtime::HeapArray<float> score_map;
    runtime::HeapArray<float> detection_maps[4];
    runtime::HeapArray<float> candidate_scores;
    runtime::HeapArray<int32_t> candidate_positions;
    runtime::HeapArray<float> sort_scores;
    runtime::HeapArray<int32_t> sort_positions;
    int capacity;
    runtime::HeapArray<float> keypoints;
    runtime::HeapArray<float> normalized;
    runtime::HeapArray<float> keypoint_scores;
    runtime::HeapArray<uint8_t> patches;
    runtime::HeapArray<uint8_t> hidden;
    runtime::HeapArray<float> sample_offsets;
    runtime::HeapArray<uint8_t> samples;
    runtime::HeapArray<uint8_t> features;
    runtime::HeapArray<float> descriptors_f32;
    runtime::HeapArray<SamplePosition> sample_positions;
    runtime::HeapArray<int> sample_order;
    runtime::HeapArray<int> sample_row_counts;
    runtime::HeapArray<hvx::Vector<float>> descriptor_scratch;
    runtime::HeapArray<float> descriptor_small_maps;
    runtime::HeapArray<float> column_weight_lanes;
    runtime::HeapArray<uint8_t> deformable_tables;
};

// Room for `bytes` and kSlack more.
template <class T>
bool allocate_with_slack(runtime::HeapArray<T> & buffer, size_t bytes) {
    return runtime::allocate(buffer, (bytes + kSlack + sizeof(T) - 1) / sizeof(T));
}

void free_keypoint_buffers(Workspace & workspace) {
    workspace.capacity = 0;
    workspace.keypoints.reset();
    workspace.normalized.reset();
    workspace.keypoint_scores.reset();
    workspace.patches.reset();
    workspace.hidden.reset();
    workspace.sample_offsets.reset();
    workspace.samples.reset();
    workspace.features.reset();
    workspace.descriptors_f32.reset();
    workspace.sample_positions.reset();
    workspace.sample_order.reset();
    workspace.sample_row_counts.reset();
}

runtime::Status allocate_workspace(Workspace & workspace, int height, int width) {
    if (workspace.height == height && workspace.width == width) return {};

    workspace = Workspace{};
    const size_t pixels = static_cast<size_t>(height) * width;
    const size_t half = pixels / 4;
    const size_t eighth = pixels / 64;
    const size_t thirty_second = pixels / 1024;
    const int factors[4] = {1, 2, 8, 32};
    workspace.image_row_bytes = (static_cast<size_t>(width) + 127) / 128 * 128 + 256;
    const size_t image_bytes = (static_cast<size_t>(height) + 2) * workspace.image_row_bytes;
    const size_t taps_bytes = static_cast<size_t>(4) * 2 * (width + height) * 4;

    bool ok = allocate_with_slack(workspace.image, image_bytes);
    ok &= allocate_with_slack(workspace.block1_mid, pixels * 32);
    ok &= allocate_with_slack(workspace.x1, pixels * 32);
    ok &= allocate_with_slack(workspace.x1_pool, half * 32);
    ok &= allocate_with_slack(workspace.block2_concatenated, half * 64);
    ok &= allocate_with_slack(workspace.x2, half * 32);
    ok &= allocate_with_slack(workspace.x2_pool, eighth * 32);
    ok &= allocate_with_slack(workspace.block3_mid, eighth * 64);
    ok &= allocate_with_slack(workspace.x3, eighth * 64);
    ok &= allocate_with_slack(workspace.x3_pool, thirty_second * 64);
    ok &= allocate_with_slack(workspace.block4_mid, thirty_second * 128);
    ok &= allocate_with_slack(workspace.x4, thirty_second * 128);
    for (int k = 0; k < 4; k++) {
        const size_t part_pixels = pixels / (static_cast<size_t>(factors[k]) * factors[k]);
        ok &= allocate_with_slack(workspace.aggregations[k], part_pixels * 32);
        ok &= allocate_with_slack(workspace.parts[k], part_pixels * 8 * 4);
        ok &= allocate_with_slack(workspace.detection_maps[k], pixels * 4);
    }
    const size_t offset_rows = eighth > thirty_second ? eighth : thirty_second;
    ok &= allocate_with_slack(workspace.offsets, offset_rows * 32 * 4);
    ok &= allocate_with_slack(workspace.dequantized, (eighth + 1) * 64 * 4);
    ok &= allocate_with_slack(workspace.deformable_tables, runtime::kHvxThreads * kDeformableLaneTableBytes +
                                                               (eighth + 31) / 32 * kDeformableTableBytes);
    const size_t column_bytes = eighth * 640 > thirty_second * 1280 ? eighth * 640 : thirty_second * 1280;
    ok &= allocate_with_slack(workspace.columns, column_bytes);
    ok &= allocate_with_slack(workspace.taps_ints, taps_bytes);
    ok &= allocate_with_slack(workspace.taps_floats, taps_bytes);
    ok &= allocate_with_slack(workspace.score_map, pixels * 4);
    ok &= allocate_with_slack(workspace.candidate_scores, pixels);
    ok &= allocate_with_slack(workspace.candidate_positions, pixels);
    ok &= allocate_with_slack(workspace.sort_scores, pixels);
    ok &= allocate_with_slack(workspace.sort_positions, pixels);
    ok &= allocate_with_slack(workspace.descriptor_scratch, static_cast<size_t>(runtime::kHvxThreads) *
                                                                kDescriptorScratchVectors * hvx::kVectorBytes);
    ok &= allocate_with_slack(workspace.descriptor_small_maps, (eighth + thirty_second) * 32 * 4);
    ok &= allocate_with_slack(workspace.column_weight_lanes, static_cast<size_t>(4) * width * 16 * 4);
    if (!ok) {
        workspace = Workspace{};
        return std::unexpected(runtime::Error::kNoMemory);
    }

    memset(workspace.image.get(), 0, image_bytes);
    for (int part = 1; part < 4; part++) {
        const size_t base = static_cast<size_t>(part) * 2 * (width + height);
        int * ints = workspace.taps_ints.get() + base;
        float * floats = workspace.taps_floats.get() + base;
        workspace.column_taps[part] = {ints, ints + width, floats, floats + width};
        workspace.row_taps[part] = {ints + 2 * width, ints + 2 * width + height, floats + 2 * width,
                                    floats + 2 * width + height};
        compute_upsample_taps(width / factors[part], width, workspace.column_taps[part]);
        compute_upsample_taps(height / factors[part], height, workspace.row_taps[part]);
        compute_column_weight_lanes(workspace.column_taps[part], width,
                                    workspace.column_weight_lanes.get() + static_cast<size_t>(part) * width * 16);
    }
    workspace.height = height;
    workspace.width = width;
    return {};
}

runtime::Status allocate_keypoint_buffers(Workspace & workspace, int max_keypoints) {
    const int capacity = (max_keypoints + 63) / 64 * 64;
    if (workspace.capacity >= capacity) return {};

    free_keypoint_buffers(workspace);
    const auto rows = static_cast<size_t>(capacity);
    bool ok = allocate_with_slack(workspace.keypoints, rows * 8);
    ok &= allocate_with_slack(workspace.normalized, rows * 8);
    ok &= allocate_with_slack(workspace.keypoint_scores, rows * 4);
    ok &= allocate_with_slack(workspace.patches, rows * 9 * 128);
    ok &= allocate_with_slack(workspace.hidden, rows * 32);
    ok &= allocate_with_slack(workspace.sample_offsets, rows * 32 * 4);
    ok &= allocate_with_slack(workspace.samples, rows * 16 * 128);
    ok &= allocate_with_slack(workspace.features, rows * 16 * 128);
    ok &= allocate_with_slack(workspace.descriptors_f32, rows * 128 * 4);
    ok &= allocate_with_slack(workspace.sample_positions, rows * 16 * sizeof(SamplePosition));
    ok &= allocate_with_slack(workspace.sample_order, rows * 16 * 4);
    ok &= allocate_with_slack(workspace.sample_row_counts,
                              static_cast<size_t>(kSortSlices) * (kMaximumDescriptorHeight + 2) * 4);
    if (!ok) {
        free_keypoint_buffers(workspace);
        return std::unexpected(runtime::Error::kNoMemory);
    }

    workspace.capacity = capacity;
    return {};
}

struct Convolution {
    int weights;
    int aux;
    const uint8_t * input;
    int height;
    int width;
    int channels;
    int taps;
    int N;
    bool selu;
    uint8_t * output;
    int output_channels;
    float * rows_f32;
    int rows_f32_columns;
};

runtime::Status run_convolution(runtime::Session & session, const LayerWeights & model, const Convolution & c) {
    convolution_int8::Arguments a;
    a.input = c.input;
    a.height = c.height;
    a.width = c.width;
    a.channels = c.channels;
    a.taps = c.taps;
    a.weights = model.part<uint8_t>(c.weights);
    a.N = c.N;

    const uint8_t * aux = c.aux >= 0 ? model.part<uint8_t>(c.aux) : nullptr;
    if (c.selu) {
        a.requantization = convolution_int8::Requantization::kSelu;
        a.selu_constants = reinterpret_cast<const float *>(aux);
        a.uint8_outputs[0] = c.output;
        a.uint8_output_channels = c.output_channels;
    } else {
        a.output_form = convolution_int8::OutputForm::kRowsF32;
        a.rows_f32 = c.rows_f32;
        a.rows_f32_columns = c.rows_f32_columns;
    }

    if (c.taps == 9) {
        const size_t constants =
            c.selu ? hvx::round_up<size_t>(static_cast<size_t>(convolution_int8::kSeluConstantCount) * c.N * 4,
                                           hvx::kVectorBytes)
                   : 0;
        a.padding = aux + constants;
    }
    return kernels::convolution_int8::run(session, a);
}

runtime::Status run_deformable(runtime::Session & session, const Workspace & workspace, const LayerWeights & model,
                               int offset_tensor, int layer_tensor, const uint8_t * input, int height, int width,
                               int channels, const uint8_t * residual, int residual_channels, int column_stride, int N,
                               uint8_t * output, int32_t * statistics) {
    const Convolution offsets{offset_tensor,
                              offset_tensor + 1,
                              input,
                              height,
                              width,
                              channels,
                              9,
                              32,
                              false,
                              nullptr,
                              0,
                              workspace.offsets.get(),
                              0};
    runtime::Status status = run_convolution(session, model, offsets);
    if (!status) return status;

    const uint8_t * padding = model.part<uint8_t>(offset_tensor + 1);
    alignas(128) float zero[128];
    for (int c = 0; c < channels; c++) zero[c] = static_cast<float>(padding[c]);

    const size_t dequantized_bytes = (static_cast<size_t>(height) * width + 1) * channels * 4;
    float * dequantized = dequantized_bytes <= session.vtcm_bytes() ? reinterpret_cast<float *>(session.vtcm())
                                                                    : workspace.dequantized.get();
    DeformableArguments sampling{input,
                                 height,
                                 width,
                                 channels,
                                 zero,
                                 workspace.offsets.get(),
                                 dequantized,
                                 workspace.deformable_tables.get(),
                                 residual,
                                 residual_channels,
                                 workspace.columns.get(),
                                 column_stride};
    runtime::StageTimer sampling_timer(statistics);
    status = run_deformable_sampling(session, sampling);
    sampling_timer.lap(stat::kDeformableSamplingMicroseconds);
    if (!status) return status;

    const Convolution layer{layer_tensor,
                            layer_tensor + 1,
                            workspace.columns.get(),
                            height,
                            width,
                            column_stride,
                            1,
                            N,
                            true,
                            output,
                            N,
                            nullptr,
                            0};
    return run_convolution(session, model, layer);
}

runtime::Status run_score_part(runtime::Session & session, const Workspace & workspace, const LayerWeights & model,
                               int k, int height, int width) {
    const Convolution part{
        tensor::kScoreHead0Part1 + k, -1, workspace.aggregations[k].get(), height, width, 32, 1, 32, false, nullptr, 0,
        workspace.parts[k].get(),     8};
    return run_convolution(session, model, part);
}

uint8_t * image_origin(const Workspace & workspace) {
    return workspace.image.get() + workspace.image_row_bytes + 128;
}

struct ImageRowsJob {
    const uint8_t * image;
    const Workspace * workspace;
};

void copy_image_rows(void * job_pointer, int thread) {
    const uint8_t * image = static_cast<const ImageRowsJob *>(job_pointer)->image;
    const Workspace & workspace = *static_cast<const ImageRowsJob *>(job_pointer)->workspace;
    const int first = workspace.height * thread / runtime::kHvxThreads;
    const int end = workspace.height * (thread + 1) / runtime::kHvxThreads;
    uint8_t * rows = image_origin(workspace);
    for (int y = first; y < end; y++) {
        memcpy(rows + static_cast<size_t>(y) * workspace.image_row_bytes,
               image + static_cast<size_t>(y) * workspace.width, static_cast<size_t>(workspace.width));
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
    Features * features = feature_slot(session, arguments.slot);
    const int model_slot = arguments.references[0];
    if (!session.model_mapped(model_slot) || height % 32 != 0 || width % 32 != 0 || height < 64 || width < 64 ||
        width > kMaximumWidth || height > kMaximumDescriptorHeight || arguments.max_keypoints < 1 ||
        features == nullptr) {
        return std::unexpected(runtime::Error::kBadParameter);
    }

    CallGuard * guard = session.state<CallGuard>();
    if (guard == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    const runtime::ExclusiveUse use(guard->in_use);
    if (!use.taken()) return std::unexpected(runtime::Error::kBadParameter);
    Workspace * kept = session.state<Workspace>();
    if (kept == nullptr) return std::unexpected(runtime::Error::kNoMemory);
    Workspace & workspace = *kept;
    runtime::Status status =
        reserve_features(features, arguments.max_keypoints, protocol::aliked::kDescriptorDimensions);
    if (!status) return status;
    status = allocate_workspace(workspace, height, width);
    if (!status) return status;
    status = allocate_keypoint_buffers(workspace, arguments.max_keypoints);
    if (!status) return status;
    features->count = 0;
    features->height = height;
    features->width = width;
    const LayerWeights model{session.model_base(model_slot), arguments.references + 1};
    const StageConstants constants(model.part<uint8_t>(tensor::kStageConstants));
    int32_t * statistics = arguments.statistics;
    runtime::StageTimer timer(statistics);
    runtime::StageTimer block1_parts(statistics);

    runtime::invalidate_host_input(arguments.image, static_cast<size_t>(height) * width);
    ImageRowsJob image_rows{arguments.image, &workspace};
    status = runtime::run_parallel(session, runtime::kHvxThreads, copy_image_rows, &image_rows);
    if (!status) return status;

    const Block1Conv1Arguments block1{
        image_origin(workspace),   workspace.image_row_bytes, height, width, model.part<uint8_t>(tensor::kBlock1Conv1),
        workspace.block1_mid.get()};
    status = run_block1_conv1(session, block1);
    if (!status) return status;
    block1_parts.lap(stat::kBlock1Conv1Microseconds);

    const Convolution block1_layers[] = {
        {tensor::kBlock1Conv2, tensor::kBlock1Conv2Aux, workspace.block1_mid.get(), height, width, 32, 9, 32, true,
         workspace.x1.get(), 32, nullptr, 0},
        {tensor::kConv1, tensor::kConv1Aux, workspace.x1.get(), height, width, 32, 1, 32, true,
         workspace.aggregations[0].get(), 32, nullptr, 0},
    };
    const int block1_statistics[2] = {stat::kBlock1Conv2Microseconds, stat::kConv1Microseconds};
    for (int i = 0; i < 2; i++) {
        status = run_convolution(session, model, block1_layers[i]);
        if (!status) return status;
        block1_parts.lap(block1_statistics[i]);
    }
    status = run_score_part(session, workspace, model, 0, height, width);
    if (!status) return status;
    block1_parts.lap(stat::kPart1Microseconds);
    timer.lap(stat::kBlock1Microseconds);

    const PoolingArguments pool1{workspace.x1.get(),
                                 height,
                                 width,
                                 32,
                                 2,
                                 constants.x1_pool_k,
                                 constants.x1_pool_z,
                                 workspace.x1_pool.get(),
                                 workspace.block2_concatenated.get()};
    status = run_average_pool(session, pool1);
    if (!status) return status;

    const Convolution block2_layers[] = {
        {tensor::kBlock2Conv1, tensor::kBlock2Conv1Aux, workspace.x1_pool.get(), height / 2, width / 2, 32, 9, 32, true,
         workspace.block2_concatenated.get(), 64, nullptr, 0},
        {tensor::kBlock2Conv2, tensor::kBlock2Conv2Aux, workspace.block2_concatenated.get(), height / 2, width / 2, 64,
         9, 32, true, workspace.x2.get(), 32, nullptr, 0},
        {tensor::kConv2, tensor::kConv2Aux, workspace.x2.get(), height / 2, width / 2, 32, 1, 32, true,
         workspace.aggregations[1].get(), 32, nullptr, 0},
    };
    runtime::StageTimer block2_parts(statistics);
    const int block2_statistics[2] = {stat::kBlock2Conv1Microseconds, stat::kBlock2Conv2Microseconds};
    for (int i = 0; i < 3; i++) {
        status = run_convolution(session, model, block2_layers[i]);
        if (!status) return status;
        if (i < 2) block2_parts.lap(block2_statistics[i]);
    }
    status = run_score_part(session, workspace, model, 1, height / 2, width / 2);
    if (!status) return status;
    timer.lap(stat::kBlock2Microseconds);

    const PoolingArguments pool2{
        workspace.x2.get(),      height / 2, width / 2, 32, 4, constants.x2_pool_k, constants.x2_pool_z,
        workspace.x2_pool.get(), nullptr};
    status = run_average_pool(session, pool2);
    if (!status) return status;

    runtime::StageTimer block3_timer(statistics);
    status = run_deformable(session, workspace, model, tensor::kBlock3Conv1Offset, tensor::kBlock3Conv1,
                            workspace.x2_pool.get(), height / 8, width / 8, 32, nullptr, 0, 384, 64,
                            workspace.block3_mid.get(), statistics);
    if (!status) return status;
    status = run_deformable(session, workspace, model, tensor::kBlock3Conv2Offset, tensor::kBlock3Conv2,
                            workspace.block3_mid.get(), height / 8, width / 8, 64, workspace.x2_pool.get(), 32, 640, 64,
                            workspace.x3.get(), statistics);
    if (!status) return status;
    block3_timer.lap(stat::kBlock3Microseconds);

    const Convolution conv3{tensor::kConv3,
                            tensor::kConv3Aux,
                            workspace.x3.get(),
                            height / 8,
                            width / 8,
                            64,
                            1,
                            32,
                            true,
                            workspace.aggregations[2].get(),
                            32,
                            nullptr,
                            0};
    status = run_convolution(session, model, conv3);
    if (!status) return status;
    status = run_score_part(session, workspace, model, 2, height / 8, width / 8);
    if (!status) return status;

    const PoolingArguments pool3{
        workspace.x3.get(),      height / 8, width / 8, 64, 4, constants.x3_pool_k, constants.x3_pool_z,
        workspace.x3_pool.get(), nullptr};
    status = run_average_pool(session, pool3);
    if (!status) return status;
    status = run_deformable(session, workspace, model, tensor::kBlock4Conv1Offset, tensor::kBlock4Conv1,
                            workspace.x3_pool.get(), height / 32, width / 32, 64, nullptr, 0, 640, 128,
                            workspace.block4_mid.get(), statistics);
    if (!status) return status;
    status = run_deformable(session, workspace, model, tensor::kBlock4Conv2Offset, tensor::kBlock4Conv2,
                            workspace.block4_mid.get(), height / 32, width / 32, 128, workspace.x3_pool.get(), 64, 1280,
                            128, workspace.x4.get(), statistics);
    if (!status) return status;

    const Convolution conv4{tensor::kConv4,
                            tensor::kConv4Aux,
                            workspace.x4.get(),
                            height / 32,
                            width / 32,
                            128,
                            1,
                            32,
                            true,
                            workspace.aggregations[3].get(),
                            32,
                            nullptr,
                            0};
    status = run_convolution(session, model, conv4);
    if (!status) return status;
    status = run_score_part(session, workspace, model, 3, height / 32, width / 32);
    if (!status) return status;
    timer.lap(stat::kDeformableMicroseconds);

    ScoreHeadArguments head{};
    head.height = height;
    head.width = width;
    for (int k = 0; k < 4; k++) {
        head.parts[k] = workspace.parts[k].get();
        head.column_taps[k] = workspace.column_taps[k];
        head.column_weight_lanes[k] = workspace.column_weight_lanes.get() + static_cast<size_t>(k) * width * 16;
        head.row_taps[k] = workspace.row_taps[k];
    }
    head.layer_tensors[0] = model.part<uint8_t>(tensor::kScoreHead2);
    head.layer_tensors[1] = model.part<uint8_t>(tensor::kScoreHead4);
    head.layer_tensors[2] = model.part<uint8_t>(tensor::kScoreHead6);
    head.score1_selu = constants.score1_selu;
    for (int k = 0; k < 3; k++) head.score_zero[k] = constants.score_zero[k];
    head.score_map = workspace.score_map.get();

    if (score_head_vtcm_bytes(width) > session.vtcm_bytes()) return std::unexpected(runtime::Error::kNoMemory);
    head.vtcm = session.vtcm();
    head.statistics = statistics;
    status = run_score_head(session, head);
    if (!status) return status;
    timer.lap(stat::kScoreHeadMicroseconds);

    DetectionArguments detection{};
    detection.score_map = workspace.score_map.get();
    detection.height = height;
    detection.width = width;
    detection.threshold = arguments.threshold;
    detection.max_keypoints = arguments.max_keypoints;

    const size_t map_bytes = static_cast<size_t>(height) * width * 4;
    if (5 * map_bytes <= session.vtcm_bytes()) {
        uint8_t * vtcm = session.vtcm();
        status = runtime::copy_memory_parallel(session, vtcm, workspace.score_map.get(), map_bytes);
        if (!status) return status;
        detection.nms_scores = reinterpret_cast<const float *>(vtcm);
        for (int k = 0; k < 4; k++) detection.maps[k] = reinterpret_cast<float *>(vtcm + (k + 1) * map_bytes);
    } else {
        detection.nms_scores = workspace.score_map.get();
        for (int k = 0; k < 4; k++) detection.maps[k] = workspace.detection_maps[k].get();
    }

    detection.candidate_scores = workspace.candidate_scores.get();
    detection.candidate_positions = workspace.candidate_positions.get();
    detection.sort_scores = workspace.sort_scores.get();
    detection.sort_positions = workspace.sort_positions.get();
    detection.keypoints = workspace.keypoints.get();
    detection.normalized = workspace.normalized.get();
    detection.scores = workspace.keypoint_scores.get();
    int count = 0;
    detection.keypoint_count = &count;
    detection.statistics = statistics;
    status = run_detection(session, detection);
    if (!status) return status;
    statistics[stat::kKeypoints] = count;
    memcpy(arguments.keypoints, workspace.keypoints.get(), static_cast<size_t>(count) * 8);
    memcpy(features->keypoints.get(), workspace.keypoints.get(), static_cast<size_t>(count) * 8);
    memcpy(arguments.scores, workspace.keypoint_scores.get(), static_cast<size_t>(count) * 4);
    timer.lap(stat::kKeypointSelectionMicroseconds);
    statistics[stat::kKeypointSelectionMicroseconds] -= statistics[stat::kNmsMicroseconds];

    if (count > 0) {
        DescriptorArguments d{};
        d.height = height;
        d.width = width;
        for (int k = 0; k < 4; k++) {
            d.aggregations[k] = workspace.aggregations[k].get();
        }
        d.column_taps = workspace.column_taps;
        d.row_taps = workspace.row_taps;
        d.constants = &constants;
        d.offset0_weights = model.part<uint8_t>(tensor::kDescriptorOffset0);
        d.offset0_aux = model.part<uint8_t>(tensor::kDescriptorOffset0Aux);
        d.offset2_weights = model.part<uint8_t>(tensor::kDescriptorOffset2);
        d.features_weights = model.part<uint8_t>(tensor::kDescriptorFeatures);
        d.features_aux = model.part<uint8_t>(tensor::kDescriptorFeaturesAux);
        d.aggregation_weights = model.part<uint8_t>(tensor::kDescriptorAggregation);
        d.keypoints = workspace.normalized.get();
        d.count = count;
        d.capacity = workspace.capacity;
        d.patches = workspace.patches.get();
        d.hidden = workspace.hidden.get();
        d.offsets = workspace.sample_offsets.get();
        d.samples = workspace.samples.get();
        d.features = workspace.features.get();
        d.descriptors_f32 = workspace.descriptors_f32.get();
        d.descriptors = features->descriptors.get();
        d.scratch = workspace.descriptor_scratch.get();
        d.small_maps = workspace.descriptor_small_maps.get();
        d.positions = workspace.sample_positions.get();
        d.sample_order = workspace.sample_order.get();
        d.row_counts = workspace.sample_row_counts.get();
        d.statistics = statistics;
        status = run_descriptors(session, d);
        if (!status) return status;
        status =
            runtime::copy_memory_parallel(session, arguments.descriptors, features->descriptors.get(),
                                          static_cast<size_t>(count) * protocol::aliked::kDescriptorDimensions * 2);
        if (!status) return status;
    }
    features->count = count;
    timer.lap(stat::kDescriptorsMicroseconds);
    return {};
}

}  // namespace benzene::models::aliked
