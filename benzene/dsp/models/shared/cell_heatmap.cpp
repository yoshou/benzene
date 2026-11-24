#include "dsp/models/shared/cell_heatmap.h"

#include <cstddef>
#include <cstdint>

#include "dsp/hvx/lane_movement.h"
#include "dsp/hvx/vector.h"
#include "dsp/vector_math/functions.h"

namespace benzene::models {
namespace {

using hvx::Mask;
using hvx::Vector;
using hvx::VectorPair;

// The softmax of the 65 logits of `count` (1 or 2) cells from `cell`: probabilities[2c], [2c + 1] = cell + c's 64
// pixels.
inline void softmax_cells(const CellHeatmapJob * job, int cell, int count, Vector<float> * probabilities) {
    const auto * logits = hvx::vectors(job->logits + static_cast<size_t>(cell) * 96);
    const Mask dustbin_lane = Mask::first_bytes(4);
    const Vector<float> minus_infinity = Vector<float>::splat_bits(0xff800000u);
    const Vector<float> log2e = Vector<float>::splat(1.4426950408889634f);
    Vector<float> x[6];
    Vector<float> maximum[2];
    Vector<float> e[6];

#pragma clang loop unroll(full)
    for (int k = 0; k < 6; k++) x[k] = k < 3 * count ? logits[k] : Vector<float>::zero();

#pragma clang loop unroll(full)
    for (int c = 0; c < 2; c++) {
        maximum[c] = max(max(x[3 * c], x[3 * c + 1]), select(dustbin_lane, x[3 * c + 2], minus_infinity));
    }
    maximum[0] = vector_math::max_to_all_lanes(maximum[0]);
    maximum[1] = vector_math::max_to_all_lanes(maximum[1]);

#pragma clang loop unroll(full)
    for (int k = 0; k < 6; k++) e[k] = x[k] - maximum[k / 3];
#pragma clang loop unroll(full)
    for (int k = 0; k < 6; k++) e[k] = e[k] * log2e;
    vector_math::exp2_batch<6>(e);
    e[2] = select(dustbin_lane, e[2], Vector<float>::zero());
    e[5] = select(dustbin_lane, e[5], Vector<float>::zero());

    Vector<float> sum[2];
    Vector<float> denominator[2];
    Vector<float> reciprocal[2];
#pragma clang loop unroll(full)
    for (int c = 0; c < 2; c++) sum[c] = e[3 * c] + e[3 * c + 1] + e[3 * c + 2];
    sum[0] = vector_math::sum_to_all_lanes(sum[0]);
    sum[1] = vector_math::sum_to_all_lanes(sum[1]);
    if (count < 2) sum[1] = Vector<float>::splat(1.0f);

    // 1 / sum: vector_math::reciprocal's steps
#pragma clang loop unroll(full)
    for (int c = 0; c < 2; c++) {
        denominator[c] = sum[c];
        reciprocal[c] = reinterpret<float>(0x7EF311C3 - reinterpret<int32_t>(denominator[c]));
    }
    for (int step = 0; step < 3; step++) {
        Vector<float> correction[2];
#pragma clang loop unroll(full)
        for (int c = 0; c < 2; c++) correction[c] = 2.0f - denominator[c] * reciprocal[c];
#pragma clang loop unroll(full)
        for (int c = 0; c < 2; c++) reciprocal[c] = reciprocal[c] * correction[c];
    }

#pragma clang loop unroll(full)
    for (int c = 0; c < 2; c++) {
        probabilities[2 * c] = e[3 * c] * reciprocal[c];
        probabilities[2 * c + 1] = e[3 * c + 1] * reciprocal[c];
    }
}

void write_heatmap_cell(const CellHeatmapJob * job, int cell_y, int cell_x, const Vector<float> * p) {
    for (int row = 0; row < 8; row++) {
        float * destination = job->heatmap + static_cast<size_t>(8 * cell_y + row) * job->stride + 8 * cell_x;
        const int offset = static_cast<int>(reinterpret_cast<uintptr_t>(destination) & 127);
        const Mask lanes = and_not(Mask::first_bytes_up_to_128(offset + 32), Mask::first_bytes(offset));
        store_masked(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(destination) & ~uintptr_t{127}), lanes,
                     rotate_right_bytes(p[row / 4], 32 * (row % 4) - offset));
    }
}

}  // namespace

// Four cells at a time where cell_x is a multiple of 4: row r of their blocks is one heatmap vector, lanes 8 (r % 4) ..
// of the four cells' p[r / 4] (a 4 x 4 transpose of 8-lane groups).
void softmax_cell_row(void * context, int cell_y, int) {
    const auto * job = static_cast<const CellHeatmapJob *>(context);
    int cell_x = 0;

    for (; cell_x + 4 <= job->cell_columns; cell_x += 4) {
        Vector<float> probabilities[8];
        softmax_cells(job, cell_y * job->cell_columns + cell_x, 2, probabilities);
        softmax_cells(job, cell_y * job->cell_columns + cell_x + 2, 2, probabilities + 4);
        for (int half = 0; half < 2; half++) {
            const VectorPair<float> a = interleave_chunks<32>(probabilities[2 + half], probabilities[half]);
            const VectorPair<float> b = interleave_chunks<32>(probabilities[6 + half], probabilities[4 + half]);
            const VectorPair<float> low = interleave_chunks<64>(b.low(), a.low());
            const VectorPair<float> high = interleave_chunks<64>(b.high(), a.high());
            float * destination = job->heatmap + static_cast<size_t>(8 * cell_y + 4 * half) * job->stride + 8 * cell_x;
            store(destination, low.low());
            store(destination + job->stride, low.high());
            store(destination + 2 * job->stride, high.low());
            store(destination + 3 * job->stride, high.high());
        }
    }

    for (; cell_x < job->cell_columns; cell_x++) {
        Vector<float> probabilities[4];
        softmax_cells(job, cell_y * job->cell_columns + cell_x, 1, probabilities);
        write_heatmap_cell(job, cell_y, cell_x, probabilities);
    }
}

}  // namespace benzene::models
