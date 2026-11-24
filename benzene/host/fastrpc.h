#pragma once

#include <cstdint>

#include "benzene.h"

namespace benzene::host {

// Call first in main, before libcdsprpc starts its threads. SIGINT, SIGTERM, SIGHUP and SIGQUIT then only set
// stop_requested(), and the commands stop at their next step: a process ended by a signal or by _exit skips
// libcdsprpc's cleanup, and the session's DSP mappings stay until the board reboots. SIGPIPE is ignored.
void install_signal_guard();
bool stop_requested();
// stop_requested(), with a message when it is.
bool stopped_by_signal();

// Pins the calling thread and the threads created after it (libcdsprpc's too) to the big cores, then opens the benzene
// session in an unsigned PD on the cDSP. Prints the reason and returns false on failure.
bool open_session();
void close_session();

remote_handle64 session();

// A benzene call's result: true for success, else false and "<call> error 0x<error>" (call: printf format, arguments).
bool rpc_succeeded(int error, const char * call, ...) __attribute__((format(printf, 2, 3)));

// An rpcmem buffer that lives until the process ends (nullptr and a message on failure).
void * allocate_rpc_memory(uint64_t bytes);

// A whole file in a new rpcmem buffer (nullptr and a message on failure); *bytes = its size.
void * load_file_into_rpc_memory(const char * path, size_t * bytes);

// File bytes [offset, offset + bytes) into an rpcmem buffer, from any thread (false and a message on failure). With
// `mapped` (the buffer is mapped into the DSP already), DMA-BUF syncs around the read write the CPU caches back.
bool fill_from_file(uint8_t * buffer, const char * path, uint64_t offset, uint64_t bytes, bool mapped);

// Registers an rpcmem buffer with FastRPC and maps it into the session's model slot `slot` (false and a message on
// failure).
bool map_into_slot(uint8_t * buffer, uint64_t bytes, int slot);
// Unmaps slot `slot` and unregisters the buffer; it keeps its contents and can be mapped again.
bool unmap_slot(uint8_t * buffer, uint64_t bytes, int slot);
// A new rpcmem buffer with file bytes [offset, offset + bytes), mapped into slot `slot`.
bool load_into_slot(const char * path, uint64_t offset, uint64_t bytes, int slot);

}  // namespace benzene::host
