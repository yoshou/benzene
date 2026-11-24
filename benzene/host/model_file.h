#pragma once

#include <cstdint>
#include <vector>

#include "host/files.h"

namespace benzene::host {

// A record of <model>.tab, the tensor table that python/benzene/model_file.py writes next to a model.
struct TensorRecord {
    char name[48];
    uint64_t offset;  // in the model file
    uint64_t bytes;
    uint32_t kind;
    uint32_t part_count;  // a layer blob's parts: byte offsets in the blob (the protocol's *_part order)
    uint32_t parts[24];
    uint32_t dimensions;
    uint32_t shape[4];
    uint32_t aux;  // per tensor kind, e.g. the lm_head's column count
};
static_assert(sizeof(TensorRecord) == 192);

class ModelTable {
  public:
    // False if the file is missing or not a table.
    bool load(const char * path);

    const TensorRecord * find(const char * name) const;
    const TensorRecord * find(const char * prefix, int index) const;  // "<prefix>.<index>"
    // The file range [*begin, *end) that holds every tensor.
    void tensor_range(uint64_t * begin, uint64_t * end) const;

  private:
    std::vector<TensorRecord> records_;
};

// The table of model.hexmodel: model.tab (false and a message: not a .hexmodel name or no table).
bool load_table_of_model(const char * model, ModelTable * table);

// Empty and a message on failure.
MallocBuffer<float> read_f32_tensor(const char * model, const TensorRecord & record);

// File bytes [offset, offset + bytes) of a model file in an rpcmem buffer.
struct WeightRange {
    const char * model = nullptr;
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint8_t * buffer = nullptr;
};

// Allocates the range's buffer (false and a message on failure).
bool allocate_range(WeightRange * range);
// Reads the range into its buffer in parallel parts (false and a message on failure).
bool read_range(const WeightRange & range);

// plan_layer_groups and layer_references read at most this many layers.
constexpr int kMaximumLayers = 64;

// Layers "<prefix>.<l>" (part_count parts each) in groups of per_group: appends a range per group to `ranges` and
// writes layer l's references {first_slot + its group, offset in the group, parts}; the last group extends to `end`
// if that lies behind its layers. Returns the layer count.
// A mapped buffer gets a DSP address (IOVA, a 32-bit space per session) at a multiple of its size rounded up to a power
// of two: per_group should make a group nearly fill such a window.
int plan_layer_groups(const ModelTable & table, const char * prefix, int part_count, int per_group, int first_slot,
                      uint64_t end, const char * model, int32_t * references, std::vector<WeightRange> * ranges);

// The references of layers "<prefix>.<l>" in one range mapped into `slot` from file offset `base`. Returns the layer
// count.
int layer_references(const ModelTable & table, const char * prefix, int slot, uint64_t base, int part_count,
                     int32_t * references);

// Allocates and reads ranges[i] and maps it into slot first_slot + i.
bool load_ranges(WeightRange * ranges, int count, int first_slot);

// Prints "model <MiB> MiB in <buffers> buffers loaded in <ms> ms".
void print_model_load(uint64_t bytes, int buffers, double microseconds);

}  // namespace benzene::host
