#pragma once

namespace benzene::models {

// logits: fp32 rows [cells][96] (65 used, lane 64 the dustbin); heatmap: [H][stride] fp32 (stride a multiple of 32).
struct CellHeatmapJob {
    const float * logits;
    float * heatmap;
    int cell_columns;
    int stride;
};

// Cell row cell_y (parallel_for item function over the cell rows): per 8 x 8 cell the softmax over its 64 pixels and
// the dustbin, the dustbin dropped, pixel k at row k / 8, column k % 8 of the cell.
void softmax_cell_row(void * context, int cell_y, int thread);

}  // namespace benzene::models
