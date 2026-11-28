// engine/metal/src/hy4flash_gguf/engram.cpp - see engram.h.
#include "engram.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include "strata/ngram/ple_reader.hpp"   // ReaderStats - engram_io.hpp only forward-declares it

namespace strata::hy4_flash {

namespace {
// The pinned STQ config's image placeholder. Sushi maps it to pad before
// compression, including when the placeholder is in a predecessor slot.
constexpr int32_t kStqImageToken = 120120;

double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

struct EngramSupply::Table {
    int block_id = -1;
    bool stq = false;
    hy4_flash::EngramHashParams p;
    hy4_flash::EngramRows reader;
    std::vector<int64_t> ids;
    std::vector<float> valid;
    std::vector<uint8_t> rows;
    std::thread th;
    std::string err;
    bool running = false, joined = false;
    double t_issue = 0, wait_us = 0;
};

EngramSupply::~EngramSupply() {
    join_all();
    for (Table* t : tables_) delete t;
    tables_.clear();
}

int EngramSupply::block_of(int table) const { return tables_[(size_t) table]->block_id; }

EngramSupply::Table* EngramSupply::find(int block_id) const {
    for (Table* t : tables_)
        if (t->block_id == block_id) return t;
    return nullptr;
}

std::string EngramSupply::open(const Model& m, const std::string& pack_dir, bool cache_pages) {
    if (ready()) return "the Engram supply is already open";
    if (m.geo.engram_tables.empty()) return "this model declares no Engram tables";
    if (m.engram_row_bytes == 0) return "the Engram table's row size is not known (an unsupported element type)";
    std::string err;
    std::vector<int32_t> token_map;
    const bool segmented = !m.engram_segments.empty();
    if (segmented) {
        if (m.engram_segments.size() != m.geo.engram_tables.size() ||
            m.engram_token_map.size() != size_t(m.geo.vocab_size) ||
            m.engram_token_map.size() <= size_t(kStqImageToken) ||
            m.geo.engram_pad_token < 0 || size_t(m.geo.engram_pad_token) >= m.engram_token_map.size())
            return "the Engram shards and token map differ from the model geometry";
        token_map = m.engram_token_map;
        token_map[kStqImageToken] = token_map[size_t(m.geo.engram_pad_token)];
    } else if (!hy4_flash::engram_token_map_load(pack_dir, token_map, err)) {
        return err;
    }
    std::vector<std::unique_ptr<Table>> opened;
    opened.reserve(m.geo.engram_tables.size());
    for (size_t table = 0; table < m.geo.engram_tables.size(); ++table) {
        const hy4_flash::EngramTable& t = m.geo.engram_tables[table];
        const EngramSegment* segment = segmented ? &m.engram_segments[table] : nullptr;
        if (segment && (segment->block_id != t.block_id || segment->rows != t.rows ||
                        t.head_offsets.size() != t.head_bucket_counts.size()))
            return "the Engram shard layout differs from the model's table geometry";
        auto e = std::make_unique<Table>();
        e->block_id = t.block_id;
        e->stq = segmented;
        e->p.ngram_size = m.geo.engram_ngram_size;
        e->p.heads_per_ngram = m.geo.engram_heads_per_ngram;
        e->p.columns = m.geo.engram_heads;
        e->p.pad_token = m.geo.engram_pad_token;
        e->p.multipliers = t.multipliers;
        e->p.primes = t.head_bucket_counts;
        e->p.offsets = t.head_offsets;
        if (segment) {
            for (size_t head = 0; head < e->p.offsets.size(); ++head) {
                const uint64_t offset = e->p.offsets[head];
                if (offset < t.row_base || offset - t.row_base > segment->rows ||
                    e->p.primes[head] > segment->rows - (offset - t.row_base)) {
                    return "an Engram bucket exceeds its shard's table";
                }
                e->p.offsets[head] = offset - t.row_base;
            }
        }
        e->p.token_map = token_map.data();
        e->p.token_map_len = (int64_t) token_map.size();
        if (!e->p.check(err)) return "Engram block " + std::to_string(t.block_id) + ": " + err;
        // Each layer has an independent reader. GGUF supplies a shared tensor with global row IDs.
        // STQ supplies a shard per layer with the offsets rebased above.
        if (!e->reader.open(segment ? segment->path : m.engram_path, segment ? segment->offset : m.engram_offset,
                            segment ? segment->rows : m.engram_rows, m.engram_row_bytes, err, cache_pages)) {
            return "Engram block " + std::to_string(t.block_id) + ": " + err;
        }
        opened.push_back(std::move(e));
    }
    tables_.reserve(opened.size());
    token_map_.swap(token_map);  // The live storage keeps every Table's token_map pointer valid.
    for (auto& table : opened) tables_.push_back(table.release());
    columns_ = m.geo.engram_heads;
    row_bytes_ = m.engram_row_bytes;
    return {};
}

std::string EngramSupply::issue(const int32_t* tokens, const int32_t* pos, const int32_t* prev, int n_tok) {
    // THE PREVIOUS WINDOW'S TAIL, charged where it belongs.  `join_all` here is not this window's work: it is a
    // reader of the last window that has not finished, so its time goes to `tail_us` (engram.h: Timing).
    const double t_tail = now_us();
    join_all();   // a caller cannot have two windows in flight: the buffers below are one window's
    tm_.tail_us += now_us() - t_tail;
    const double t_hash = now_us();
    n_tok_ = n_tok;
    std::string err;
    for (Table* t : tables_) {
        // The hash stays on this thread: it is a few hundred multiplications, it is the part that must be
        // bit-exact (engram.hpp lists four ways to get it wrong), and keeping it here means the worker does
        // nothing but read.
        if (!hy4_flash::engram_row_ids(t->p, tokens, pos, prev, nullptr, n_tok, t->ids, t->valid, err))
            return "Engram block " + std::to_string(t->block_id) + ": " + err;
        if (t->stq) {
            for (int i = 0; i < n_tok; ++i) {
                const int32_t token = tokens[i];
                if (token == t->p.pad_token || token == kStqImageToken)
                    std::fill_n(t->valid.data() + size_t(i) * columns_, columns_, 0.f);
            }
        }
        t->rows.assign(t->ids.size() * (size_t) row_bytes_, 0);
        t->err.clear();
        t->joined = false;
        t->running = true;
        t->t_issue = now_us();
        Table* raw = t;
        raw->th = std::thread([raw] {
            std::string e;
            if (!raw->reader.gather(raw->ids.data(), raw->ids.size(), raw->rows.data(), e)) raw->err = e;
        });
    }
    tm_.hash_us += now_us() - t_hash;
    ++tm_.windows;
    return {};
}

const uint8_t* EngramSupply::join(int block_id, std::string& err) {
    Table* t = find(block_id);
    if (t == nullptr) { err = "block " + std::to_string(block_id) + " has no Engram table"; return nullptr; }
    if (t->running) {
        const double t0 = now_us();
        t->th.join();
        t->running = false;
        t->joined = true;
        t->wait_us = now_us() - t0;
        tm_.join_us += t->wait_us;
        ++tm_.joins;
    }
    if (!t->joined) { err = "the Engram rows of block " + std::to_string(block_id) + " were never issued"; return nullptr; }
    if (!t->err.empty()) { err = t->err; return nullptr; }
    return t->rows.data();
}

const float* EngramSupply::valid(int block_id) const {
    const Table* t = find(block_id);
    // Deliberately only after the join: a caller that reads the rows before `join` would get zeros, and a caller
    // that reads the mask before `join` would get the previous window's.  Both are silent wrong answers, so the
    // mask is gated on the same handshake as the rows (engram.h's wait-point contract).
    return t != nullptr && t->joined ? t->valid.data() : nullptr;
}

EngramSupply::IoStats EngramSupply::io_stats() const {
    IoStats a;
    double p50 = 0, p99 = 0;
    int n = 0;
    for (const Table* t : tables_) {
        const ngram::ReaderStats* s = t->reader.stats();
        if (s == nullptr) continue;
        a.requests += s->requests;
        a.cache_hits += s->cache_hits;
        a.dedup_rows += s->dedup_rows;
        a.reads += s->reads;
        a.bytes += s->bytes;
        a.wait_us += s->wait_us;
        a.read_us_sum += s->read_us_sum;
        // The percentiles are per reader and cannot be summed; averaged over the tables that have any reads,
        // which is what a window's wait is made of (it waits for all four).
        if (s->reads > 0) {
            p50 += s->percentile(0.50);
            p99 += s->percentile(0.99);
            ++n;
        }
    }
    if (n > 0) { a.p50_us = p50 / n; a.p99_us = p99 / n; }
    return a;
}

double EngramSupply::wait_us(int block_id) const {
    const Table* t = find(block_id);
    return t != nullptr ? t->wait_us : 0.0;
}

// Deliberately NOT charged to Timing::join_us: the only caller inside a window is `issue`, which charges this
// to `tail_us` (it is the previous window's reader, not this window's wait), and the destructor's call is not a
// window at all.  Charging here as well would count the same microseconds twice.
void EngramSupply::join_all() {
    for (Table* t : tables_)
        if (t->running) {
            const double t0 = now_us();
            t->th.join();
            t->running = false;
            t->joined = true;
            t->wait_us = now_us() - t0;
        }
}

}  // namespace strata::hy4_flash
