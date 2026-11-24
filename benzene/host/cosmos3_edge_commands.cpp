#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "host/command_line.h"
#include "host/commands.h"
#include "host/fastrpc.h"
#include "host/files.h"
#include "host/model_file.h"
#include "host/timing.h"
#include "benzene/protocol.h"

namespace benzene::host {
namespace {

namespace decode_stat = protocol::cosmos3_edge::llm_decode_stat;
namespace head_reference = protocol::cosmos3_edge::llm_head_reference;
namespace prefill_stat = protocol::cosmos3_edge::llm_prefill_stat;
namespace vision_stat = protocol::cosmos3_edge::vision_stat;
using protocol::cosmos3_edge::kLlmHidden;
using protocol::cosmos3_edge::kLlmLayer4BitReferences;
using protocol::cosmos3_edge::kLlmLayerReferences;
using protocol::cosmos3_edge::kMaximumVisionImages;
using protocol::cosmos3_edge::kPatchesPerVisionToken;
using protocol::cosmos3_edge::kRopeValuesPerToken;
using protocol::cosmos3_edge::kVisionHidden;
using protocol::cosmos3_edge::kVisionLayerReferences;
using protocol::cosmos3_edge::kVisionTailReferences;

// Layers per rpcmem buffer: a group of layers fits one 256 MiB IOVA window.
constexpr int kLlmLayersPerBuffer = 5;
constexpr int kVisionLayersPerBuffer = 16;
constexpr int k4BitLayersPerBuffer = 9;
constexpr int kMaximum4BitReferences = kLlmLayer4BitReferences * kMaximumLayers + 2;
constexpr int kKvCacheMargin = 32;
constexpr int kVisionSlot = 0;
constexpr int kDefaultMaxNewTokens = 256;
constexpr int kMaximumNewTokens = 1 << 16;
constexpr int kRopeAngles = kRopeValuesPerToken / 2;

class BackgroundThread {
  public:
    BackgroundThread() = default;
    BackgroundThread(const BackgroundThread &) = delete;
    BackgroundThread & operator=(const BackgroundThread &) = delete;
    ~BackgroundThread() { join(); }

    template <typename Function>
    void start(Function function) {
        thread_ = std::thread(function);
    }
    void join() {
        if (thread_.joinable()) thread_.join();
    }

  private:
    std::thread thread_;
};

// The 4-bit decode weights: per LLM layer a blob "llm.dec4.<l>" of four Q4_0 GEMV streams (qkv, o, fc1, fc2), and the
// stream "llm.dec4.head". Appends their ranges (the head with the last) to `ranges` for the slots from first_slot and
// writes the decode call's dec4 argument to references_4bit; returns its length (0 and a message on error).
int plan_4bit_decode(const ModelTable & table, const char * model, int layers, int columns, int first_slot,
                     int32_t * references_4bit, std::vector<WeightRange> * ranges) {
    const TensorRecord * head = table.find("llm.dec4.head");
    if (head == nullptr || table.find("llm.dec4.0") == nullptr) {
        fprintf(stderr, "--dec4: %s has no 4-bit decode weights (export_hexmodel --dec4)\n", model);
        return 0;
    }

    const size_t first_group = ranges->size();
    const int layers_4bit =
        plan_layer_groups(table, "llm.dec4", protocol::cosmos3_edge::llm_layer_4bit_part::kCount, k4BitLayersPerBuffer,
                          first_slot, head->offset + head->bytes, model, references_4bit, ranges);
    const int groups = static_cast<int>(ranges->size() - first_group);
    if (layers_4bit != layers || static_cast<int>(head->aux) != columns ||
        first_slot + groups > protocol::kModelSlots || head->offset < ranges->back().offset) {
        fprintf(stderr, "--dec4: unexpected 4-bit weight layout (%d layers, %d groups)\n", layers_4bit, groups);
        return 0;
    }

    references_4bit[kLlmLayer4BitReferences * layers_4bit] = first_slot + groups - 1;
    references_4bit[kLlmLayer4BitReferences * layers_4bit + 1] =
        static_cast<int32_t>(head->offset - ranges->back().offset);
    return kLlmLayer4BitReferences * layers_4bit + 2;
}

// The end of the last LLM layer group, which holds the final norm and, unless the decode is 4-bit, the int8 lm_head.
uint64_t llm_head_end(const TensorRecord & norm, const TensorRecord & lm_head, const TensorRecord & lm_head_scales,
                      bool decode_4bit) {
    uint64_t end = lm_head_scales.offset + lm_head_scales.bytes;
    if (lm_head.offset + lm_head.bytes > end) end = lm_head.offset + lm_head.bytes;
    if (norm.offset + norm.bytes > end) end = norm.offset + norm.bytes;
    return decode_4bit ? norm.offset + norm.bytes : end;
}

// The decode step's head (protocol::cosmos3_edge::llm_head_reference), all in slot `slot` mapped from `base`.
void plan_head(const TensorRecord & norm, const TensorRecord & lm_head, const TensorRecord & lm_head_scales, int slot,
               uint64_t base, int32_t * head) {
    head[head_reference::kNormSlot] = slot;
    head[head_reference::kNormOffset] = static_cast<int32_t>(norm.offset - base);
    head[head_reference::kLmHeadSlot] = slot;
    head[head_reference::kLmHeadOffset] = static_cast<int32_t>(lm_head.offset - base);
    head[head_reference::kScalesSlot] = slot;
    head[head_reference::kScalesOffset] = static_cast<int32_t>(lm_head_scales.offset - base);
    head[head_reference::kColumns] = static_cast<int32_t>(lm_head.aux);
}

// cos / sin of a text position (all three M-RoPE sections equal), in f32 like torch:
// angle i = position * theta^(-2i / 128).
void text_rope(float position, float * cosines, float * sines) {
    for (int i = 0; i < kRopeAngles; i++) {
        const auto inverse_frequency = static_cast<float>(1.0 / pow(1e8, (2.0 * i) / 128.0));
        const float angle = position * inverse_frequency;
        cosines[i] = cosf(angle);
        sines[i] = sinf(angle);
    }
}

bool embedding_row(FILE * embeddings, int token, float * row) {
    _Float16 values[kLlmHidden];
    if (fseeko(embeddings, static_cast<off_t>(token) * kLlmHidden * 2, SEEK_SET) != 0 ||
        fread(values, 2, kLlmHidden, embeddings) != kLlmHidden) {
        return false;
    }
    for (int i = 0; i < kLlmHidden; i++) row[i] = static_cast<float>(values[i]);
    return true;
}

int most_likely_token(const float * logits, int count) {
    int best = 0;
    for (int i = 1; i < count; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

struct PromptBundle {
    const float * vision_input = nullptr;  // rpcmem
    MallocBuffer<int32_t> image_boundaries;
    MallocBuffer<int32_t> prompt_ids;
    MallocBuffer<float> prompt_rope;
    int images = 0;
    int patches = 0;
    int prompt_tokens = 0;
    int rope_delta = 0;
    int image_token = 0;
    int video_token = 0;
    std::vector<int> end_of_sequence_ids;
};

bool read_prompt_bundle(const char * directory, PromptBundle * bundle) {
    size_t vision_bytes = 0;
    size_t boundary_bytes = 0;
    size_t id_bytes = 0;
    size_t rope_bytes = 0;
    size_t metadata_bytes = 0;
    char path[512];
    const auto path_of = [&](const char * name) {
        snprintf(path, sizeof path, "%s/%s", directory, name);
        return path;
    };

    bundle->image_boundaries = load_file<int32_t>(path_of("image_boundaries.i32"), &boundary_bytes);
    bundle->prompt_ids = load_file<int32_t>(path_of("prompt_ids.i32"), &id_bytes);
    bundle->prompt_rope = load_file<float>(path_of("prompt_rope.f32"), &rope_bytes);
    const MallocBuffer<int32_t> metadata = load_file<int32_t>(path_of("metadata.i32"), &metadata_bytes);
    bundle->vision_input =
        static_cast<const float *>(load_file_into_rpc_memory(path_of("vision_input.f32"), &vision_bytes));
    if (!bundle->image_boundaries || !bundle->prompt_ids || !bundle->prompt_rope || !metadata ||
        bundle->vision_input == nullptr) {
        fprintf(stderr, "generate: cannot read the bundle %s\n", directory);
        return false;
    }

    constexpr int kMetadataFields = 5;
    if (metadata_bytes < kMetadataFields * 4 ||
        metadata_bytes != static_cast<size_t>(kMetadataFields + metadata[4]) * 4) {
        fprintf(stderr, "bad bundle metadata\n");
        return false;
    }
    bundle->prompt_tokens = metadata[0];
    bundle->rope_delta = metadata[1];
    bundle->image_token = metadata[2];
    bundle->video_token = metadata[3];
    bundle->end_of_sequence_ids.assign(metadata.get() + kMetadataFields,
                                       metadata.get() + kMetadataFields + metadata[4]);

    const int32_t * boundaries = bundle->image_boundaries.get();
    bundle->images = static_cast<int>(boundary_bytes / 4) - 1;
    bundle->patches = bundle->images > 0 ? boundaries[bundle->images] : 0;
    const auto prompt_tokens = static_cast<size_t>(bundle->prompt_tokens);
    if (boundary_bytes < 4 || boundaries[0] != 0 || id_bytes != prompt_tokens * 4 ||
        rope_bytes != prompt_tokens * kRopeValuesPerToken * 4 ||
        vision_bytes != static_cast<size_t>(bundle->patches) * kVisionHidden * 4 ||
        bundle->patches % kPatchesPerVisionToken != 0) {
        fprintf(stderr, "bad bundle\n");
        return false;
    }
    return true;
}

bool embed_prompt(const PromptBundle & bundle, FILE * embeddings, const float * vision_tokens, float * x) {
    const int vision_token_count = bundle.patches / kPatchesPerVisionToken;
    int used_vision_tokens = 0;

    for (int position = 0; position < bundle.prompt_tokens; position++) {
        float * row = x + static_cast<size_t>(position) * kLlmHidden;
        const int token = bundle.prompt_ids[position];
        if (token != bundle.image_token && token != bundle.video_token) {
            if (!embedding_row(embeddings, token, row)) {
                fprintf(stderr, "read embedding of token %d failed\n", token);
                return false;
            }
            continue;
        }

        if (used_vision_tokens >= vision_token_count) {
            fprintf(stderr, "more image tokens than vision tokens\n");
            return false;
        }
        memcpy(row, vision_tokens + static_cast<size_t>(used_vision_tokens) * kLlmHidden, kLlmHidden * 4);
        used_vision_tokens++;
    }

    if (used_vision_tokens != vision_token_count) {
        fprintf(stderr, "image tokens %d != vision tokens %d\n", used_vision_tokens, vision_token_count);
        return false;
    }
    return true;
}

struct GenerationWeights {
    // By slot: the LLM layer groups (the final norm and the lm_head with the last), the vision layer groups (the tail
    // with the last), then the 4-bit decode groups.
    std::vector<WeightRange> ranges;
    std::vector<int32_t> llm_references = std::vector<int32_t>(kMaximumLayers * kLlmLayerReferences);
    std::vector<int32_t> vision_references = std::vector<int32_t>(kMaximumLayers * kVisionLayerReferences);
    int32_t vision_tail_references[kVisionTailReferences] = {};
    int32_t head[head_reference::kCount] = {};
    int32_t references_4bit[kMaximum4BitReferences] = {};
    int references_4bit_count = 0;
    int llm_layers = 0;
    int vision_layers = 0;
    int head_slot = 0;
    int first_vision_slot = 0;
    int slots = 0;  // the LLM and vision ranges
    int groups_4bit = 0;
    // 4-bit ranges mapped from the start; the others take the vision weights' windows after the vision run (the DSP's
    // address space does not hold all the weights together with the KV cache and the workspaces).
    int early_4bit = 0;
    int columns = 0;  // vocabulary
};

bool plan_generation_weights(const ModelTable & table, const char * model, bool decode_4bit, GenerationWeights * w) {
    const TensorRecord * lm_head_scales = table.find("llm.lm_head.s");
    const TensorRecord * lm_head = table.find("llm.lm_head");
    const TensorRecord * norm = table.find("llm.norm_g");
    const TensorRecord * vision_tail = table.find("vision.tail");
    if (table.find("llm.layers.0") == nullptr || lm_head_scales == nullptr || lm_head == nullptr || norm == nullptr ||
        table.find("vision.layers.0") == nullptr || vision_tail == nullptr) {
        fprintf(stderr, "missing tensors\n");
        return false;
    }

    w->llm_layers = plan_layer_groups(
        table, "llm.layers", protocol::cosmos3_edge::llm_layer_part::kCount, kLlmLayersPerBuffer, 0,
        llm_head_end(*norm, *lm_head, *lm_head_scales, decode_4bit), model, w->llm_references.data(), &w->ranges);
    w->head_slot = static_cast<int>(w->ranges.size()) - 1;
    w->first_vision_slot = w->head_slot + 1;
    const uint64_t head_base = w->llm_layers > 0 ? w->ranges[w->head_slot].offset : 0;
    w->vision_layers = plan_layer_groups(
        table, "vision.layers", protocol::cosmos3_edge::vision_layer_part::kCount, kVisionLayersPerBuffer,
        w->first_vision_slot, vision_tail->offset + vision_tail->bytes, model, w->vision_references.data(), &w->ranges);
    w->slots = static_cast<int>(w->ranges.size());
    if (w->llm_layers == 0 || w->vision_layers == 0 || w->slots > protocol::kModelSlots ||
        vision_tail->offset < w->ranges[w->slots - 1].offset || norm->offset < head_base ||
        lm_head->offset < head_base || lm_head_scales->offset < head_base) {
        fprintf(stderr, "unexpected weight layout\n");
        return false;
    }

    w->vision_tail_references[0] = w->slots - 1;
    w->vision_tail_references[1] = static_cast<int32_t>(vision_tail->offset - w->ranges[w->slots - 1].offset);
    for (int i = 0; i < protocol::cosmos3_edge::vision_tail_part::kCount; i++) {
        w->vision_tail_references[2 + i] = static_cast<int32_t>(vision_tail->parts[i]);
    }

    w->columns = static_cast<int>(lm_head->aux);
    plan_head(*norm, *lm_head, *lm_head_scales, w->head_slot, head_base, w->head);
    if (!decode_4bit) return true;

    w->references_4bit_count =
        plan_4bit_decode(table, model, w->llm_layers, w->columns, w->slots, w->references_4bit, &w->ranges);
    if (w->references_4bit_count == 0) return false;
    w->head[head_reference::kLmHeadOffset] = 0;
    w->head[head_reference::kScalesOffset] = 0;
    w->groups_4bit = static_cast<int>(w->ranges.size()) - w->slots;
    const int vision_groups = w->slots - w->first_vision_slot;
    w->early_4bit = w->groups_4bit > vision_groups ? w->groups_4bit - vision_groups : 0;
    return true;
}

int run_vision_images(const GenerationWeights & weights, const PromptBundle & bundle, int first_image,
                      float * vision_tokens, int32_t * statistics) {
    const int images = std::min(bundle.images - first_image, kMaximumVisionImages);
    const int first_patch = bundle.image_boundaries[first_image];
    const int patches = bundle.image_boundaries[first_image + images] - first_patch;
    int32_t boundaries[kMaximumVisionImages + 1];
    for (int i = 0; i <= images; i++) boundaries[i] = bundle.image_boundaries[first_image + i] - first_patch;

    return benzene_cosmos3_edge_vision_encoder(
        session(), weights.vision_references.data(), weights.vision_layers * kVisionLayerReferences,
        weights.vision_tail_references, kVisionTailReferences,
        bundle.vision_input + static_cast<size_t>(first_patch) * kVisionHidden, patches * kVisionHidden, boundaries,
        images + 1, vision_tokens + static_cast<size_t>(first_patch / kPatchesPerVisionToken) * kLlmHidden,
        patches / kPatchesPerVisionToken * kLlmHidden, statistics, vision_stat::kCount);
}

// The rpcmem buffers of GenerationWeights. start() maps all of them but the late 4-bit ones, the largest first: a
// mapping made after the DSP heap has grown may find no IOVA window. Host threads then read the LLM buffers, then the
// 4-bit ones; each late one is allocated (late_4bit_allocated_), mapped after the vision run (late_4bit_mapped_), read.
class WeightLoader {
  public:
    explicit WeightLoader(GenerationWeights & weights) : weights_(weights) {}
    WeightLoader(const WeightLoader &) = delete;
    WeightLoader & operator=(const WeightLoader &) = delete;
    ~WeightLoader() {
        stopping_ = true;
        language_model_reader_.join();
        reader_4bit_.join();
    }

    bool start() {
        std::vector<WeightRange> & ranges = weights_.ranges;
        bool vision_read = true;
        BackgroundThread vision_reader;
        vision_reader.start([&] {
            for (int slot = weights_.first_vision_slot; slot < weights_.slots; slot++) {
                vision_read = allocate_range(&ranges[slot]) && read_range(ranges[slot]) && vision_read;
            }
        });

        bool allocated = true;
        for (int slot = 0; slot < first_late_4bit() && allocated; slot++) {
            if (slot >= weights_.first_vision_slot && slot < weights_.slots) continue;
            allocated = allocate_range(&ranges[slot]);
        }
        vision_reader.join();
        if (!allocated || !vision_read) return false;

        const WeightRange & head_range = ranges[weights_.head_slot];
        if (!map_into_slot(head_range.buffer, head_range.bytes, weights_.head_slot)) return false;
        for (int slot = 0; slot < first_late_4bit(); slot++) {
            if (slot == weights_.head_slot) continue;
            if (!map_into_slot(ranges[slot].buffer, ranges[slot].bytes, slot)) return false;
        }

        language_model_reader_.start([this] { read_language_model(); });
        if (weights_.groups_4bit > 0) reader_4bit_.start([this] { read_4bit(); });
        return true;
    }

    bool wait_for_language_model() {
        language_model_reader_.join();
        return !language_model_failed_;
    }

    bool swap_4bit_into_vision_windows() {
        std::vector<WeightRange> & ranges = weights_.ranges;
        for (int slot = weights_.first_vision_slot; slot < weights_.slots; slot++) {
            if (!unmap_slot(ranges[slot].buffer, ranges[slot].bytes, slot)) return false;
        }

        while (!late_4bit_allocated_) usleep(100);
        for (int slot = first_late_4bit(); slot < end_4bit(); slot++) {
            if (!map_into_slot(ranges[slot].buffer, ranges[slot].bytes, slot)) return false;
        }
        late_4bit_mapped_ = true;
        return true;
    }

    bool wait_for_4bit() {
        reader_4bit_.join();
        return !failed_4bit_;
    }

    bool swap_vision_back() {
        std::vector<WeightRange> & ranges = weights_.ranges;
        for (int slot = first_late_4bit(); slot < end_4bit(); slot++) {
            if (!unmap_slot(ranges[slot].buffer, ranges[slot].bytes, slot)) return false;
        }
        for (int slot = weights_.first_vision_slot; slot < weights_.slots; slot++) {
            if (!map_into_slot(ranges[slot].buffer, ranges[slot].bytes, slot)) return false;
        }
        return true;
    }

  private:
    int first_late_4bit() const { return weights_.slots + weights_.early_4bit; }
    int end_4bit() const { return weights_.slots + weights_.groups_4bit; }

    void read_language_model() {
        bool failed = false;
        for (int slot = 0; slot < weights_.first_vision_slot && !stopping_; slot++) {
            const WeightRange & range = weights_.ranges[slot];
            failed = !fill_from_file(range.buffer, range.model, range.offset, range.bytes, true) || failed;
        }
        language_model_failed_ = failed;
        language_model_done_ = true;
    }

    void read_4bit() {
        std::vector<WeightRange> & ranges = weights_.ranges;
        const int first_late = first_late_4bit();
        const int end = end_4bit();
        bool failed = false;

        for (int slot = first_late; slot < end; slot++) failed = !allocate_range(&ranges[slot]) || failed;
        late_4bit_allocated_ = true;

        while (!language_model_done_ && !stopping_) usleep(200);
        for (int slot = weights_.slots; slot < end && !failed && !stopping_; slot++) {
            while (slot >= first_late && !late_4bit_mapped_ && !stopping_) usleep(200);
            if (stopping_) break;
            failed =
                !fill_from_file(ranges[slot].buffer, ranges[slot].model, ranges[slot].offset, ranges[slot].bytes, true);
        }
        failed_4bit_ = failed || stopping_;
    }

    GenerationWeights & weights_;
    BackgroundThread language_model_reader_;
    BackgroundThread reader_4bit_;
    std::atomic<bool> language_model_failed_{false};
    std::atomic<bool> language_model_done_{false};
    std::atomic<bool> failed_4bit_{false};
    std::atomic<bool> late_4bit_allocated_{false};
    std::atomic<bool> late_4bit_mapped_{false};
    std::atomic<bool> stopping_{false};
};

// The language model's layers in groups of kLlmLayersPerBuffer from slot 0; the last group extends to `end`. The
// layer count (0 and a message: none).
int plan_language_model(const ModelTable & table, const char * model, uint64_t end, std::vector<int32_t> * references,
                        std::vector<WeightRange> * ranges) {
    references->resize(static_cast<size_t>(kMaximumLayers) * kLlmLayerReferences);
    const int layers = plan_layer_groups(table, "llm.layers", protocol::cosmos3_edge::llm_layer_part::kCount,
                                         kLlmLayersPerBuffer, 0, end, model, references->data(), ranges);
    if (layers == 0) fprintf(stderr, "missing tensor llm.layers.0\n");
    return layers;
}

// <directory>/<name> as Ts (empty and a message on failure); *count = the number of values.
template <typename T>
MallocBuffer<T> load_bundle_file(const char * directory, const char * name, size_t * count) {
    const std::string path = join_path(directory, name);
    size_t bytes = 0;
    MallocBuffer<T> values = load_file<T>(path.c_str(), &bytes);
    if (!values || bytes % sizeof(T) != 0) {
        fprintf(stderr, "bad %s\n", path.c_str());
        return nullptr;
    }
    *count = bytes / sizeof(T);
    return values;
}

}  // namespace

int run_cosmos3_edge_llm_prefill(int argc, char ** argv) {
    if (!parse_command_line(argc, argv, 3, nullptr, 0, "model.hexmodel bundle_dir out_dir")) return 1;
    const char * model = argv[2];
    const char * bundle = argv[3];
    const char * output = argv[4];
    ModelTable table;
    if (!load_table_of_model(model, &table) || !make_directory(output)) return 1;

    size_t x_values = 0;
    size_t rope_values = 0;
    const MallocBuffer<float> x = load_bundle_file<float>(bundle, "x.f32", &x_values);
    const MallocBuffer<float> rope = load_bundle_file<float>(bundle, "rope.f32", &rope_values);
    if (!x || !rope) return 1;
    const auto tokens = static_cast<int>(x_values / kLlmHidden);
    if (tokens < 1 || x_values != static_cast<size_t>(tokens) * kLlmHidden ||
        rope_values != static_cast<size_t>(tokens) * kRopeValuesPerToken) {
        fprintf(stderr, "bad bundle %s: x.f32 [tokens][%d], rope.f32 [tokens][%d]\n", bundle, kLlmHidden,
                kRopeValuesPerToken);
        return 1;
    }

    std::vector<int32_t> references;
    std::vector<WeightRange> ranges;
    const int layers = plan_language_model(table, model, 0, &references, &ranges);
    const double load_start = now_microseconds();
    if (layers == 0 || !load_ranges(ranges.data(), static_cast<int>(ranges.size()), 0)) return 1;
    print_model_load(ranges.back().offset + ranges.back().bytes - ranges[0].offset, static_cast<int>(ranges.size()),
                     now_microseconds() - load_start);

    const MallocBuffer<float> y = allocate_buffer<float>(x_values);
    if (y == nullptr) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    int32_t statistics[prefill_stat::kCount] = {};
    const double start = now_microseconds();
    const int error = benzene_cosmos3_edge_llm_prefill(
        session(), references.data(), layers * kLlmLayerReferences, x.get(), static_cast<int>(x_values), rope.get(),
        static_cast<int>(rope_values), static_cast<uint32_t>(tokens), y.get(), static_cast<int>(x_values), statistics,
        prefill_stat::kCount);
    if (!rpc_succeeded(error, "cosmos3_edge_llm_prefill")) return 1;

    printf("llm prefill %d layers, %d tokens: %.1f ms (call %.1f) | ", layers, tokens,
           statistics[prefill_stat::kLayersMicroseconds] / 1e3, (now_microseconds() - start) / 1e3);
    print_block_stages(statistics);
    printf("\n");
    return write_file(output, "y.f32", y.get(), x_values * 4) ? 0 : 1;
}

int run_cosmos3_edge_llm_decode(int argc, char ** argv) {
    bool decode_4bit = false;
    const Option options[] = {{"--dec4", nullptr, &decode_4bit}};
    if (!parse_command_line(argc, argv, 3, options, 1, "model.hexmodel bundle_dir out_dir [--dec4]")) return 1;
    const char * model = argv[2];
    const char * bundle = argv[3];
    const char * output = argv[4];
    ModelTable table;
    if (!load_table_of_model(model, &table) || !make_directory(output)) return 1;
    const TensorRecord * norm = table.find("llm.norm_g");
    const TensorRecord * lm_head = table.find("llm.lm_head");
    const TensorRecord * lm_head_scales = table.find("llm.lm_head.s");
    if (norm == nullptr || lm_head == nullptr || lm_head_scales == nullptr) {
        fprintf(stderr, "missing tensor llm.norm_g, llm.lm_head or llm.lm_head.s\n");
        return 1;
    }

    size_t x_values = 0;
    size_t rope_values = 0;
    const MallocBuffer<float> x = load_bundle_file<float>(bundle, "x.f32", &x_values);
    const MallocBuffer<float> rope_rows = load_bundle_file<float>(bundle, "rope.f32", &rope_values);
    if (!x || !rope_rows) return 1;
    const std::string meta_path = join_path(bundle, "meta.txt");
    const File meta = open_file(meta_path.c_str(), "r");
    int tokens = 0;
    int generated = 0;
    const auto rows = static_cast<int>(x_values / kLlmHidden);
    if (!meta || fscanf(meta.get(), "%d %d", &tokens, &generated) != 2 || tokens < 1 || generated < 1 ||
        tokens + generated - 1 > rows || rope_values != static_cast<size_t>(rows) * kRopeValuesPerToken) {
        fprintf(stderr, "bad bundle %s: meta.txt \"prompt_tokens answer_tokens\", x.f32 and rope.f32 rows\n", bundle);
        return 1;
    }

    std::vector<int32_t> references;
    std::vector<WeightRange> ranges;
    const int layers = plan_language_model(table, model, llm_head_end(*norm, *lm_head, *lm_head_scales, decode_4bit),
                                           &references, &ranges);
    if (layers == 0) return 1;
    const int llm_groups = static_cast<int>(ranges.size());
    const int head_slot = llm_groups - 1;
    const uint64_t head_base = ranges[head_slot].offset;
    if (norm->offset < head_base || lm_head->offset < head_base || lm_head_scales->offset < head_base) {
        fprintf(stderr, "the head tensors are not in the last layer group\n");
        return 1;
    }
    const auto columns = static_cast<int>(lm_head->aux);
    int32_t head[head_reference::kCount];
    plan_head(*norm, *lm_head, *lm_head_scales, head_slot, head_base, head);
    int32_t references_4bit[kMaximum4BitReferences];
    int references_4bit_count = 0;
    if (decode_4bit) {
        references_4bit_count = plan_4bit_decode(table, model, layers, columns, llm_groups, references_4bit, &ranges);
        if (references_4bit_count == 0) return 1;
        head[head_reference::kLmHeadOffset] = 0;
        head[head_reference::kScalesOffset] = 0;
    }
    const int groups_4bit = static_cast<int>(ranges.size()) - llm_groups;

    // The largest range first: its IOVA window is the hardest to find.
    const double load_start = now_microseconds();
    if (!load_ranges(&ranges[head_slot], 1, head_slot) || !load_ranges(ranges.data(), head_slot, 0) ||
        !load_ranges(ranges.data() + llm_groups, groups_4bit, llm_groups)) {
        return 1;
    }
    print_model_load(ranges.back().offset + ranges.back().bytes - ranges[0].offset, static_cast<int>(ranges.size()),
                     now_microseconds() - load_start);
    printf("decode weights: %s\n", decode_4bit ? "4-bit (--dec4)" : "int8");

    const float * cosines = rope_rows.get();
    const float * sines = rope_rows.get() + static_cast<size_t>(rows) * kRopeAngles;
    if (!rpc_succeeded(benzene_cosmos3_edge_kv_cache(session(), static_cast<uint32_t>(layers),
                                                     static_cast<uint32_t>(tokens + generated + kKvCacheMargin)),
                       "cosmos3_edge_kv_cache")) {
        return 1;
    }

    std::vector<float> rope(static_cast<size_t>(tokens) * kRopeValuesPerToken);
    std::vector<float> y(static_cast<size_t>(tokens) * kLlmHidden);
    std::vector<float> logits(static_cast<size_t>(generated) * columns);
    memcpy(rope.data(), cosines, static_cast<size_t>(tokens) * kRopeAngles * 4);
    memcpy(rope.data() + static_cast<size_t>(tokens) * kRopeAngles, sines,
           static_cast<size_t>(tokens) * kRopeAngles * 4);
    int32_t statistics[prefill_stat::kCount] = {};
    double start = now_microseconds();
    if (!rpc_succeeded(benzene_cosmos3_edge_llm_prefill(
                           session(), references.data(), layers * kLlmLayerReferences, x.get(), tokens * kLlmHidden,
                           rope.data(), tokens * kRopeValuesPerToken, static_cast<uint32_t>(tokens), y.data(),
                           tokens * kLlmHidden, statistics, prefill_stat::kCount),
                       "cosmos3_edge_llm_prefill")) {
        return 1;
    }
    const double prefill_time = now_microseconds() - start;

    float token_rope[kRopeValuesPerToken] = {};
    int32_t decode_statistics[decode_stat::kCount] = {};
    if (!rpc_succeeded(
            benzene_cosmos3_edge_llm_decode_step(
                session(), references.data(), 0, head, head_reference::kCount, references_4bit, references_4bit_count,
                y.data() + static_cast<size_t>(tokens - 1) * kLlmHidden, kLlmHidden, token_rope, kRopeValuesPerToken,
                logits.data(), columns, decode_statistics, decode_stat::kCount),
            "cosmos3_edge_llm_decode_step (head)")) {
        return 1;
    }

    start = now_microseconds();
    for (int i = 1; i < generated; i++) {
        if (stopped_by_signal()) return 1;
        const int row = tokens + i - 1;
        memcpy(token_rope, cosines + static_cast<size_t>(row) * kRopeAngles, kRopeAngles * 4);
        memcpy(token_rope + kRopeAngles, sines + static_cast<size_t>(row) * kRopeAngles, kRopeAngles * 4);
        if (!rpc_succeeded(
                benzene_cosmos3_edge_llm_decode_step(
                    session(), references.data(), layers * kLlmLayerReferences, head, head_reference::kCount,
                    references_4bit, references_4bit_count, x.get() + static_cast<size_t>(row) * kLlmHidden, kLlmHidden,
                    token_rope, kRopeValuesPerToken, logits.data() + static_cast<size_t>(i) * columns, columns,
                    decode_statistics, decode_stat::kCount),
                "cosmos3_edge_llm_decode_step (token %d)", i)) {
            return 1;
        }
    }
    const double decode_time = now_microseconds() - start;

    printf("llm prefill %d tokens: %.1f ms (call %.1f) | ", tokens, statistics[prefill_stat::kLayersMicroseconds] / 1e3,
           prefill_time / 1e3);
    print_block_stages(statistics);
    printf("\n");
    if (generated > 1) {
        const int32_t * s = decode_statistics;
        printf(
            "llm decode %d tokens: %.2f ms per token (calls) | last token: norm1+qkv %.2f rope+append %.2f attention "
            "%.2f output %.2f norm2+fc1 %.2f fc2 %.2f head %.2f\n",
            generated - 1, decode_time / 1e3 / (generated - 1), s[decode_stat::kNormAndQkvMicroseconds] / 1e3,
            s[decode_stat::kRopeAndAppendMicroseconds] / 1e3, s[decode_stat::kAttentionMicroseconds] / 1e3,
            s[decode_stat::kOutputMicroseconds] / 1e3, s[decode_stat::kNorm2AndFc1Microseconds] / 1e3,
            s[decode_stat::kFc2Microseconds] / 1e3, s[decode_stat::kHeadMicroseconds] / 1e3);
    }
    return write_file(output, "logits.f32", logits.data(), logits.size() * 4) ? 0 : 1;
}

int run_cosmos3_edge_vision(int argc, char ** argv) {
    if (!parse_command_line(argc, argv, 3, nullptr, 0, "model.hexmodel bundle_dir out_dir")) return 1;
    const char * model = argv[2];
    const char * bundle = argv[3];
    const char * output = argv[4];
    ModelTable table;
    if (!load_table_of_model(model, &table) || !make_directory(output)) return 1;
    const TensorRecord * first_layer = table.find("vision.layers.0");
    const TensorRecord * tail = table.find("vision.tail");
    if (first_layer == nullptr || tail == nullptr) {
        fprintf(stderr, "missing tensor vision.layers.0 or vision.tail\n");
        return 1;
    }

    size_t x_values = 0;
    size_t offset_count = 0;
    const MallocBuffer<float> x = load_bundle_file<float>(bundle, "x.f32", &x_values);
    const MallocBuffer<int32_t> image_offsets = load_bundle_file<int32_t>(bundle, "image_offsets.i32", &offset_count);
    if (!x || !image_offsets) return 1;
    const int images = static_cast<int>(offset_count) - 1;
    if (images < 1 || images > kMaximumVisionImages || image_offsets[0] != 0 ||
        x_values != static_cast<size_t>(image_offsets[images]) * kVisionHidden) {
        fprintf(stderr, "bad bundle %s: image_offsets.i32 of 1 .. %d images, x.f32 [patches][%d]\n", bundle,
                kMaximumVisionImages, kVisionHidden);
        return 1;
    }
    const int patches = image_offsets[images];

    const uint64_t begin = first_layer->offset;
    const uint64_t end = tail->offset + tail->bytes;
    std::vector<int32_t> references(static_cast<size_t>(kMaximumLayers) * kVisionLayerReferences);
    const int layers = layer_references(table, "vision.layers", kVisionSlot, begin,
                                        protocol::cosmos3_edge::vision_layer_part::kCount, references.data());
    int32_t tail_references[kVisionTailReferences] = {kVisionSlot, static_cast<int32_t>(tail->offset - begin)};
    for (int i = 0; i < protocol::cosmos3_edge::vision_tail_part::kCount; i++) {
        tail_references[2 + i] = static_cast<int32_t>(tail->parts[i]);
    }
    const double load_start = now_microseconds();
    if (!load_into_slot(model, begin, end - begin, kVisionSlot)) return 1;
    print_model_load(end - begin, 1, now_microseconds() - load_start);

    const size_t y_values = static_cast<size_t>(patches / kPatchesPerVisionToken) * kLlmHidden;
    const MallocBuffer<float> y = allocate_buffer<float>(y_values);
    if (y == nullptr) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    int32_t statistics[vision_stat::kCount] = {};
    const double start = now_microseconds();
    const int error = benzene_cosmos3_edge_vision_encoder(
        session(), references.data(), layers * kVisionLayerReferences, tail_references, kVisionTailReferences, x.get(),
        static_cast<int>(x_values), image_offsets.get(), images + 1, y.get(), static_cast<int>(y_values), statistics,
        vision_stat::kCount);
    if (!rpc_succeeded(error, "cosmos3_edge_vision_encoder")) return 1;

    printf("vision %d layers + tail, %d images / %d patches: %.0f ms (call %.0f) | ", layers, images, patches,
           (statistics[vision_stat::kLayersMicroseconds] + statistics[vision_stat::kTailMicroseconds]) / 1e3,
           (now_microseconds() - start) / 1e3);
    print_block_stages(statistics);
    printf(" tail %.1f | workspace %.1f setup %.1f\n", statistics[vision_stat::kTailMicroseconds] / 1e3,
           statistics[vision_stat::kWorkspaceMicroseconds] / 1e3,
           statistics[vision_stat::kGroupSetupMicroseconds] / 1e3);
    return write_file(output, "y.f32", y.get(), y_values * 4) ? 0 : 1;
}

int run_cosmos3_edge_generate(int argc, char ** argv) {
    bool decode_4bit = false;
    const char * max_new_tokens_text = nullptr;
    const Option options[] = {{"--dec4", nullptr, &decode_4bit}, {"--max-new-tokens", &max_new_tokens_text}};
    if (!parse_command_line(argc, argv, 3, options, 2,
                            "model.hexmodel bundle_dir out_dir [--dec4] [--max-new-tokens N]")) {
        return 1;
    }
    int max_new_tokens = kDefaultMaxNewTokens;
    if (max_new_tokens_text != nullptr &&
        !parse_integer(max_new_tokens_text, "--max-new-tokens", 1, kMaximumNewTokens, &max_new_tokens)) {
        return 1;
    }
    const char * model = argv[2];
    const char * output = argv[4];
    ModelTable table;
    if (!load_table_of_model(model, &table) || !make_directory(output)) return 1;
    const std::string model_path(model);
    const std::string embeddings_path = model_path.substr(0, model_path.find_last_of('/') + 1) + "embed_tokens.f16";
    const File embeddings = open_file(embeddings_path.c_str(), "rb");
    if (!embeddings) {
        fprintf(stderr, "missing %s\n", embeddings_path.c_str());
        return 1;
    }

    const double run_start = now_microseconds();
    GenerationWeights weights;
    if (!plan_generation_weights(table, model, decode_4bit, &weights)) return 1;
    WeightLoader loader(weights);
    if (!loader.start()) return 1;
    double load_time = now_microseconds() - run_start;
    PromptBundle bundle;
    if (!read_prompt_bundle(argv[3], &bundle)) return 1;

    const size_t vision_token_bytes = static_cast<size_t>(bundle.patches / kPatchesPerVisionToken) * kLlmHidden * 4;
    auto * vision_tokens = static_cast<float *>(allocate_rpc_memory(vision_token_bytes > 0 ? vision_token_bytes : 128));
    if (vision_tokens == nullptr) return 1;
    const auto kv_capacity = static_cast<uint32_t>(bundle.prompt_tokens + max_new_tokens + kKvCacheMargin);
    int kv_error = 0;
    BackgroundThread kv_thread;
    kv_thread.start([&] {
        kv_error = benzene_cosmos3_edge_kv_cache(session(), static_cast<uint32_t>(weights.llm_layers), kv_capacity);
        if (kv_error != 0) return;
        kv_error = benzene_cosmos3_edge_prepare_workspaces(session(), static_cast<uint32_t>(bundle.prompt_tokens));
    });

    int32_t vision_statistics[vision_stat::kCount] = {};
    double start = now_microseconds();
    int error = 0;
    for (int first_image = 0; first_image < bundle.images; first_image += kMaximumVisionImages) {
        error = run_vision_images(weights, bundle, first_image, vision_tokens, vision_statistics);
        if (error != 0) break;
    }
    const double vision_time = now_microseconds() - start;
    kv_thread.join();
    if (!rpc_succeeded(error, "cosmos3_edge_vision_encoder") ||
        !rpc_succeeded(kv_error, "cosmos3_edge_kv_cache / cosmos3_edge_prepare_workspaces") || stopped_by_signal()) {
        return 1;
    }

    start = now_microseconds();
    if (decode_4bit && !loader.swap_4bit_into_vision_windows()) return 1;
    const double swap_in_time = now_microseconds() - start;
    if (!loader.wait_for_language_model()) return 1;
    const double load_after_vision = now_microseconds() - start;
    load_time += load_after_vision;

    start = now_microseconds();
    auto * x = static_cast<float *>(allocate_rpc_memory(static_cast<size_t>(bundle.prompt_tokens) * kLlmHidden * 4));
    if (x == nullptr || !embed_prompt(bundle, embeddings.get(), vision_tokens, x)) return 1;
    const double embedding_time = now_microseconds() - start;

    float last_hidden[kLlmHidden];
    std::vector<float> logits(static_cast<size_t>(weights.columns));
    float token_rope[kRopeValuesPerToken] = {};
    int32_t prefill_statistics[prefill_stat::kCount] = {};
    int32_t decode_statistics[decode_stat::kCount] = {};
    start = now_microseconds();
    error = benzene_cosmos3_edge_llm_prefill(
        session(), weights.llm_references.data(), weights.llm_layers * kLlmLayerReferences, x,
        bundle.prompt_tokens * kLlmHidden, bundle.prompt_rope.get(), bundle.prompt_tokens * kRopeValuesPerToken,
        static_cast<uint32_t>(bundle.prompt_tokens), last_hidden, kLlmHidden, prefill_statistics, prefill_stat::kCount);
    double read_wait_4bit = 0;
    if (weights.groups_4bit > 0) {
        const double wait_start = now_microseconds();
        if (!loader.wait_for_4bit()) return 1;
        read_wait_4bit = now_microseconds() - wait_start;
    }
    if (!rpc_succeeded(error, "cosmos3_edge_llm_prefill")) return 1;
    error = benzene_cosmos3_edge_llm_decode_step(
        session(), weights.llm_references.data(), 0, weights.head, head_reference::kCount, weights.references_4bit,
        weights.references_4bit_count, last_hidden, kLlmHidden, token_rope, kRopeValuesPerToken, logits.data(),
        weights.columns, decode_statistics, decode_stat::kCount);
    if (!rpc_succeeded(error, "cosmos3_edge_llm_decode_step (head)")) return 1;
    const double prefill_time = now_microseconds() - start;
    if (!write_file(output, "first_logits.f32", logits.data(), logits.size() * 4)) return 1;

    const auto is_end_of_sequence = [&bundle](int token) {
        for (const int id : bundle.end_of_sequence_ids) {
            if (token == id) return true;
        }
        return false;
    };
    std::vector<int32_t> generated;
    generated.push_back(most_likely_token(logits.data(), weights.columns));
    float token_embedding[kLlmHidden];
    start = now_microseconds();
    while (static_cast<int>(generated.size()) < max_new_tokens && !is_end_of_sequence(generated.back()) &&
           !stop_requested()) {
        const int sequence_index = bundle.prompt_tokens + static_cast<int>(generated.size()) - 1;
        if (!embedding_row(embeddings.get(), generated.back(), token_embedding)) return 1;
        text_rope(static_cast<float>(sequence_index + bundle.rope_delta), token_rope, token_rope + kRopeAngles);
        error = benzene_cosmos3_edge_llm_decode_step(
            session(), weights.llm_references.data(), weights.llm_layers * kLlmLayerReferences, weights.head,
            head_reference::kCount, weights.references_4bit, weights.references_4bit_count, token_embedding, kLlmHidden,
            token_rope, kRopeValuesPerToken, logits.data(), weights.columns, decode_statistics, decode_stat::kCount);
        if (!rpc_succeeded(error, "cosmos3_edge_llm_decode_step (token %d)", static_cast<int>(generated.size()))) {
            return 1;
        }
        generated.push_back(most_likely_token(logits.data(), weights.columns));
    }
    const double decode_time = now_microseconds() - start;
    const int decode_steps = static_cast<int>(generated.size()) - 1;
    if (!write_file(output, "generated_ids.i32", generated.data(), generated.size() * 4)) return 1;

    start = now_microseconds();
    if (decode_4bit && !loader.swap_vision_back()) return 1;
    const double swap_back_time = now_microseconds() - start;

    if (decode_4bit) {
        printf(
            "decode weights: 4-bit (--dec4; %d buffers, %d of them in the vision weights' windows: swapped in %.0f ms, "
            "back %.0f ms; read wait after the prefill %.0f ms)\n",
            weights.groups_4bit, weights.groups_4bit - weights.early_4bit, swap_in_time / 1e3, swap_back_time / 1e3,
            read_wait_4bit / 1e3);
    } else {
        printf("decode weights: int8\n");
    }
    const double total = now_microseconds() - run_start;
    printf(
        "load %.1f s (after vision %.2f) | vision %.2f s (%d images, %d patches) | prefill %.2f s (%d tokens) | "
        "decode %d tokens %.2f s (%.1f ms/token, %s) | between stages %.2f s (prompt rows %.2f) | total %.2f s\n",
        load_time / 1e6, load_after_vision / 1e6, vision_time / 1e6, bundle.images, bundle.patches, prefill_time / 1e6,
        bundle.prompt_tokens, decode_steps, decode_time / 1e6,
        decode_steps > 0 ? decode_time / 1e3 / decode_steps : 0.0,
        is_end_of_sequence(generated.back()) ? "ended"
        : stop_requested()                   ? "stopped by a signal"
                                             : "token limit",
        (total - load_time - vision_time - prefill_time - decode_time) / 1e6, embedding_time / 1e6, total / 1e6);
    return stopped_by_signal() ? 1 : 0;
}

}  // namespace benzene::host
