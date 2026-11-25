#include <vector>

#include "host/command_line.h"
#include "host/commands.h"
#include "host/fastrpc.h"
#include "host/files.h"
#include "host/model_file.h"
#include "host/vggt_network.h"
#include "benzene/protocol.h"

namespace benzene::host {

int run_vggt(int argc, char ** argv) {
    bool keep_tokens = false;
    const Option options[] = {{"--tokens", nullptr, &keep_tokens}};
    if (!parse_command_line(argc, argv, 3, options, 1, "model.hexmodel bundle_dir out_dir [--tokens]")) return 1;
    const char * model = argv[2];
    const char * output = argv[4];

    ModelTable table;
    if (!load_table_of_model(model, &table) || !make_directory(output)) return 1;
    VggtBundle bundle;
    if (!read_vggt_bundle(argv[3], &bundle)) return 1;
    const int frames = bundle.frames;
    const int height = bundle.height;
    const int width = bundle.width;
    const int patch_rows = height / protocol::vggt::kPatchSize;
    const int patch_columns = width / protocol::vggt::kPatchSize;
    const int tokens = frames * (protocol::vggt::kSpecialTokens + patch_rows * patch_columns);
    VggtWeights weights;
    int32_t dpt[2][protocol::vggt::kDptReferences];
    const DptHeadTensors heads[2] = {
        {"vggt.dpt.depth", "misc", protocol::vggt::dpt_head_part::kCount, dpt[0]},
        {"vggt.dpt.point", "misc", protocol::vggt::dpt_head_part::kCount, dpt[1]},
    };
    if (!plan_vggt_weights(table, model, &weights) || !plan_dpt_heads(table, model, heads, 2, &weights)) {
        return 1;
    }

    const std::vector<float> uv = dpt_uv_tables(patch_rows, patch_columns, height, width);
    _Float16 * patches = nullptr;
    std::vector<float> grid_positions;
    float special[protocol::vggt::kSpecialTokens * protocol::vggt::kHidden];
    if (!load_vggt_weights(model, table, bundle, &weights, &patches, &grid_positions, special)) return 1;

    const size_t kept_values =
        keep_tokens ? static_cast<size_t>(protocol::vggt::kKeptOutputs) * tokens * protocol::vggt::kKeptOutputWidth : 0;
    auto * kept = keep_tokens ? static_cast<_Float16 *>(allocate_rpc_memory(kept_values * 2)) : nullptr;
    if (keep_tokens && kept == nullptr) return 1;
    if (!run_vggt_transformer("vggt", weights, bundle, patches, grid_positions, special, kept, kept_values)) return 1;
    if (keep_tokens && !write_file(output, "inter.f16", kept, kept_values * 2)) return 1;
    if (stopped_by_signal()) return 1;

    std::vector<float> pose;
    if (!run_vggt_camera_head(weights, frames, &pose)) return 1;
    if (!write_file(output, "pose.f32", pose.data(), pose.size() * 4)) return 1;

    static const char * const kHeads[2] = {"depth", "point"};
    static const int kHeadChannels[2] = {2, 4};
    static const char * const kHeadFiles[2] = {"depth.f32", "points.f32"};
    for (int head = 0; head < 2; head++) {
        if (stopped_by_signal()) return 1;
        const size_t values = static_cast<size_t>(frames) * height * width * kHeadChannels[head];
        auto * out = static_cast<float *>(allocate_rpc_memory(values * 4));
        if (out == nullptr) return 1;
        if (!run_vggt_dpt_head(kHeads[head], dpt[head], kHeadChannels[head], uv, out, values)) return 1;
        if (!write_file(output, kHeadFiles[head], out, values * 4)) return 1;
    }
    return 0;
}

}  // namespace benzene::host
