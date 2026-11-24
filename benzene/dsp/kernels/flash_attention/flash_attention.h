#pragma once

#include <cstdint>

#include "dsp/runtime/session.h"
#include "dsp/runtime/status.h"

namespace benzene::kernels::flash_attention {

// Where Q, K and V come from.
enum class Source : uint8_t {
    // Q / O token-major [token][head][D] fp16 (128-byte aligned), K / V token-major [key][kv head][D]. K / V are packed
    // into the caller's tiles (tiles set) or the kept KV tiles first.
    kRows,
    // keys [0, keys) of a packed KV cache (tiles); Q / O token-major
    kKeyValueCache,
    // tiles the qkv GEMM wrote (gemm_w8a8 kAttentionTiles*): Q^T weight tiles query_tiles
    // [head][queries / 32][d tiles], K tiles [head][queries / 32][d tiles], V^T tiles [head][d tiles][queries / 32]
    // (d >= D zero). The padding tokens may hold any finite values. Segments only.
    kTiles,
    // Q / K / V / O rows in VTCM above the session's VTCM use (row_layout; D a multiple of 32). K may be Q. O may
    // overwrite K
    // when K is not Q: the K / V rows are packed before the first O row is written. O's padding columns are written as
    // zeros. Segments only.
    kRowsInVtcm,
};

// Packed K / V^T tiles of `capacity` key tiles per KV head: a KV cache, or the caller's buffers for the packed rows
// (kRows: heads * keys / 32 * ceil(D / 32) tiles of 2 KB, 4 KB aligned).
//   K    [kv head][key tile][d tile]  activation tiles, rows = keys
//   V^T  [kv head][d tile][key tile]  activation tiles of V^T, rows = d
// fp16 tiles of 32 x 32: activation (and output) byte (r / 2) * 128 + c * 4 + (r % 2) * 2, weight byte
// (k / 2) * 128 + c * 4 + (k % 2) * 2.
struct KeyValueTiles {
    void * key_tiles = nullptr;
    void * value_tiles = nullptr;
    int capacity = 0;
};

// Packed images: segment i = tokens [first_tokens[i], + token_counts[i]) of the token-major rows (first tokens
// multiples of 32, rows between images zero), non-causal attention within each segment.
struct Segments {
    int count = 0;
    const int * first_tokens = nullptr;
    const int * token_counts = nullptr;
    const int * key_segments =
        nullptr;  // segment i's queries attend to segment key_segments[i]'s keys (nullptr: its own)
};

// Token rows in VTCM: token_stride halfwords from one token's row to the next, head h's D values from h * head_stride
// on (head_stride >= D a multiple of 64: whole vectors per head; the columns of a head beyond D are padding).
struct RowLayout {
    int token_stride = 0;
    int head_stride = 0;
};

struct Arguments {
    Source source = Source::kRows;
    int queries = 0;                      // Nq (segments: the padded tokens, a multiple of 32, at most 16384)
    int keys = 0;                         // Nk (segments: = queries)
    int query_heads = 0;                  // Hq
    int kv_heads = 0;                     // Hkv (Hq a multiple; segments: = Hq)
    int head_dimensions = 0;              // D <= 128, a multiple of 8
    bool causal = false;                  // query t sees keys <= t
    const Segments * segments = nullptr;  // packed images (not causal)
    _Float16 * q = nullptr;
    _Float16 * k = nullptr;
    _Float16 * v = nullptr;
    _Float16 * o = nullptr;
    KeyValueTiles tiles{};               // kKeyValueCache, kTiles; kRows: optional
    const void * query_tiles = nullptr;  // kTiles
    RowLayout row_layout{};              // kRowsInVtcm
};

// O = softmax(Q K^T / sqrt(D)) V per query head, query head h with KV head h / (Hq / Hkv). The DMA reads past the
// caches: this call flushes the data cache first. One call at a time (a second caller gets kBadParameter).
runtime::Status run(runtime::Session & session, const Arguments & arguments);

// Writes one key / value row ([kv heads][D] fp16) at key position `position` of a KV cache.
void append_to_cache(const _Float16 * key, const _Float16 * value, int position, int kv_heads, int head_dimensions,
                     const KeyValueTiles & cache);

}  // namespace benzene::kernels::flash_attention
