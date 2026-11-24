#include "host/files.h"

#include <sys/stat.h>

#include <cerrno>
#include <cstdio>

namespace benzene::host {

std::string join_path(const char * directory, const char * name) {
    return std::string(directory) + "/" + name;
}

bool file_size(FILE * file, size_t * bytes) {
    if (fseeko(file, 0, SEEK_END) != 0) return false;
    *bytes = static_cast<size_t>(ftello(file));
    return fseeko(file, 0, SEEK_SET) == 0;
}

void * read_file(const char * path, size_t * bytes) {
    const File file = open_file(path, "rb");
    if (!file || !file_size(file.get(), bytes)) return nullptr;
    MallocBuffer<uint8_t> data = allocate_buffer<uint8_t>(*bytes);
    if (data && fread(data.get(), 1, *bytes, file.get()) != *bytes) data.reset();
    return data.release();
}

bool make_directory(const char * path) {
    if (mkdir(path, 0755) == 0 || errno == EEXIST) return true;
    fprintf(stderr, "cannot create %s\n", path);
    return false;
}

bool write_file(const char * path, const void * data, size_t bytes) {
    const File file = open_file(path, "wb");
    if (!file || fwrite(data, 1, bytes, file.get()) != bytes) {
        fprintf(stderr, "write %s failed\n", path);
        return false;
    }
    return true;
}

bool write_file(const char * directory, const char * name, const void * data, size_t bytes) {
    return write_file(join_path(directory, name).c_str(), data, bytes);
}

}  // namespace benzene::host
