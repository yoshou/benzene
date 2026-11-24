#include "host/model_file.h"

#include <cstdio>
#include <cstring>
#include <thread>

#include "host/fastrpc.h"

namespace benzene::host {

bool ModelTable::load(const char * path) {
    constexpr size_t kHeaderBytes = 8;  // "HXTB", then the record count (int32)
    size_t bytes = 0;
    const MallocBuffer<uint8_t> data = load_file(path, &bytes);
    if (data == nullptr || bytes < kHeaderBytes || memcmp(data.get(), "HXTB", 4) != 0) return false;

    int32_t count;
    memcpy(&count, data.get() + 4, 4);
    if (count < 0 || bytes != kHeaderBytes + static_cast<size_t>(count) * sizeof(TensorRecord)) return false;

    records_.resize(static_cast<size_t>(count));
    memcpy(records_.data(), data.get() + kHeaderBytes, static_cast<size_t>(count) * sizeof(TensorRecord));
    return true;
}

const TensorRecord * ModelTable::find(const char * name) const {
    for (const TensorRecord & record : records_) {
        if (strcmp(record.name, name) == 0) return &record;
    }
    return nullptr;
}

const TensorRecord * ModelTable::find(const char * prefix, int index) const {
    char name[64];
    snprintf(name, sizeof name, "%s.%d", prefix, index);
    return find(name);
}

void ModelTable::tensor_range(uint64_t * begin, uint64_t * end) const {
    *begin = ~0ull;
    *end = 0;
    for (const TensorRecord & record : records_) {
        if (record.offset < *begin) *begin = record.offset;
        if (record.offset + record.bytes > *end) *end = record.offset + record.bytes;
    }
}

bool load_table_of_model(const char * model, ModelTable * table) {
    constexpr char kExtension[] = ".hexmodel";
    constexpr size_t kExtensionLength = sizeof kExtension - 1;
    const size_t length = strlen(model);
    if (length <= kExtensionLength || strcmp(model + length - kExtensionLength, kExtension) != 0) {
        fprintf(stderr, "%s: not a .hexmodel file\n", model);
        return false;
    }
    char path[512];
    snprintf(path, sizeof path, "%.*s.tab", static_cast<int>(length - kExtensionLength), model);
    if (!table->load(path)) {
        fprintf(stderr, "bad table %s\n", path);
        return false;
    }
    return true;
}

MallocBuffer<float> read_f32_tensor(const char * model, const TensorRecord & record) {
    MallocBuffer<float> values = allocate_buffer<float>(record.bytes / 4);
    const File file = open_file(model, "rb");
    if (values == nullptr || !file || fseeko(file.get(), static_cast<off_t>(record.offset), SEEK_SET) != 0 ||
        fread(values.get(), 1, record.bytes, file.get()) != record.bytes) {
        fprintf(stderr, "read %s failed\n", record.name);
        return nullptr;
    }
    return values;
}

bool allocate_range(WeightRange * range) {
    range->buffer = static_cast<uint8_t *>(allocate_rpc_memory(range->bytes));
    return range->buffer != nullptr;
}

bool read_range(const WeightRange & range) {
    constexpr int kParts = 4;
    constexpr uint64_t kPageMask = ~static_cast<uint64_t>(4095);
    bool part_read[kParts] = {};
    std::thread threads[kParts];
    for (int part = 0; part < kParts; part++) {
        threads[part] = std::thread([&range, part, &part_read] {
            const uint64_t begin = (range.bytes * part / kParts) & kPageMask;
            const uint64_t end = part + 1 == kParts ? range.bytes : (range.bytes * (part + 1) / kParts) & kPageMask;
            part_read[part] =
                fill_from_file(range.buffer + begin, range.model, range.offset + begin, end - begin, false);
        });
    }

    bool read = true;
    for (int part = 0; part < kParts; part++) {
        threads[part].join();
        read = read && part_read[part];
    }
    return read;
}

int plan_layer_groups(const ModelTable & table, const char * prefix, int part_count, int per_group, int first_slot,
                      uint64_t end, const char * model, int32_t * references, std::vector<WeightRange> * ranges) {
    const int stride = part_count + 2;
    const size_t first_group = ranges->size();

    int layers = 0;
    for (; layers < kMaximumLayers; layers++) {
        const TensorRecord * record = table.find(prefix, layers);
        if (record == nullptr) break;

        if (layers % per_group == 0) {
            WeightRange group;
            group.model = model;
            group.offset = record->offset;
            ranges->push_back(group);
        }
        WeightRange & group = ranges->back();
        group.bytes = record->offset + record->bytes - group.offset;

        int32_t * reference = references + layers * stride;
        reference[0] = first_slot + static_cast<int>(ranges->size() - first_group) - 1;
        reference[1] = static_cast<int32_t>(record->offset - group.offset);
        for (int i = 0; i < part_count; i++) reference[2 + i] = static_cast<int32_t>(record->parts[i]);
    }

    if (ranges->size() > first_group) {
        WeightRange & last = ranges->back();
        if (end > last.offset + last.bytes) last.bytes = end - last.offset;
    }
    return layers;
}

int layer_references(const ModelTable & table, const char * prefix, int slot, uint64_t base, int part_count,
                     int32_t * references) {
    int layers = 0;
    for (; layers < kMaximumLayers; layers++) {
        const TensorRecord * record = table.find(prefix, layers);
        if (record == nullptr) break;
        int32_t * reference = references + layers * (part_count + 2);
        reference[0] = slot;
        reference[1] = static_cast<int32_t>(record->offset - base);
        for (int i = 0; i < part_count; i++) reference[2 + i] = static_cast<int32_t>(record->parts[i]);
    }
    return layers;
}

bool load_ranges(WeightRange * ranges, int count, int first_slot) {
    for (int i = 0; i < count; i++) {
        if (!allocate_range(&ranges[i]) || !read_range(ranges[i]) ||
            !map_into_slot(ranges[i].buffer, ranges[i].bytes, first_slot + i)) {
            return false;
        }
    }
    return true;
}

void print_model_load(uint64_t bytes, int buffers, double microseconds) {
    printf("model %.1f MiB in %d buffer%s loaded in %.0f ms\n", bytes / 1048576.0, buffers, buffers != 1 ? "s" : "",
           microseconds / 1e3);
}

}  // namespace benzene::host
