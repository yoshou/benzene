#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

namespace benzene::host {

struct FreeDeleter {
    void operator()(void * pointer) const { free(pointer); }
};
template <typename T>
using MallocBuffer = std::unique_ptr<T[], FreeDeleter>;

template <typename T>
MallocBuffer<T> allocate_buffer(size_t count) {
    return MallocBuffer<T>(static_cast<T *>(malloc(count != 0 ? count * sizeof(T) : 1)));
}

struct FileCloser {
    void operator()(FILE * file) const { fclose(file); }
};
using File = std::unique_ptr<FILE, FileCloser>;

inline File open_file(const char * path, const char * mode) {
    return File(fopen(path, mode));
}

// "<directory>/<name>"
std::string join_path(const char * directory, const char * name);

// The size of an open file, which is left at its start (false if it cannot seek).
bool file_size(FILE * file, size_t * bytes);

// The whole file in malloc'd memory (empty if it cannot be read); *bytes = its size.
void * read_file(const char * path, size_t * bytes);

template <typename T = uint8_t>
MallocBuffer<T> load_file(const char * path, size_t * bytes) {
    return MallocBuffer<T>(static_cast<T *>(read_file(path, bytes)));
}

// Creates the directory unless it exists (false and a message on failure).
bool make_directory(const char * path);

// Prints a message and returns false on failure.
bool write_file(const char * path, const void * data, size_t bytes);
bool write_file(const char * directory, const char * name, const void * data, size_t bytes);

}  // namespace benzene::host
