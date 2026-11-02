// src/dsv41/ds4_stage.hpp - DeepSeek-V4.1 split across processes: the link between two stages.
//
// Each stage is a self-contained slice of the model (its blocks renumbered from 0, its KV and index source lists
// rebased). The LAST stage serves the requests; the stage before it (--stage-serve HOST:PORT) answers it over a
// socket with the hand-off a layer's next group would take: h | pre of the rows it ran. The first layer of the next
// stage reads `pre` where a whole model would keep it from the previous layer's FFN.
//
// topk and cand do not travel: a slice only starts at a layer that is a KV, index and candidate source, which writes
// them itself. The Engram layers stay in the first stage, the only one with token ids to hash.
//
// The state follows the serving stage call by call, so both hold the same positions and checkpoints:
//   'H' token                     -> u32 row_floats
//   'Z'                           -> (reset)
//   'K' i64 slot                  -> (checkpoint)
//   'N'                           -> (pin)
//   'B' i64 limit                 -> u8 restored
//   'W' i64 pos0, u32 T, i32 tok[T] -> f32 rows[T * row_floats]   (a window, or a prompt chunk of up to 512 rows:
//                                      every h, then every pre, so that a frame stays under the relay's 64 MiB)
// Frames are the POM's envelope of stage_link.hpp; an answer starts with a status byte (0 ok, 1 error + message) and
// its rows go as a tensor, so the node between the machines carries them in the deployment's hidden-state precision.
//
// The DSpark drafter stays on the last stage: it reads layers 37-39 and borrows the head. A verification window comes
// as 'V', and 'T' takes its rejected rows back, as on the serving stage:
//   'V' i64 pos0, u32 T, i32 tok[T] -> f32 rows[T * row_floats]
//   'T' i64 (T << 32 | keep)        -> (the window's first `keep` rows stay)
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../program/stage_link.hpp"
#include "ds4_model.hpp"

namespace strata::ds4::stage {
namespace sl = strata::stage;

inline bool token_matches(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    unsigned char diff = 0;
    for (size_t i = 0; i < a.size(); ++i) diff |= (unsigned char) (a[i] ^ b[i]);
    return diff == 0;
}

/// The earlier stage's side: answers one connection at a time until the process ends.
inline int serve(Model& m, const std::string& addr, const std::string& token) {
    std::string err;
    int bound = 0;
    const int lfd = sl::tcp_open(addr, true, err, &bound);
    if (lfd < 0) {
        std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
        return 1;
    }
    const size_t row = m.stage_row_floats();
    // (the line a supervisor waits for, as the other engine's stage server prints it: the port it is reached on)
    std::fprintf(stderr, "maya-ds4: stage listening on %s (port %d), %zu floats a row\n", addr.c_str(), bound, row);
    std::fflush(stderr);
    std::printf("READY stage\n");
    std::fflush(stdout);
    std::vector<char> req, ans;
    std::vector<int> toks;
    std::vector<float> rows;
    for (;;) {
        const int fd = ::accept(lfd, nullptr, nullptr);
        if (fd < 0) continue;
        bool authed = false;
        while (sl::recv_msg(fd, req)) {
            if (req.empty()) break;
            const char op = req[0];
            size_t at = 1, tensor_from = 0;
            bool ok = true;
            std::string why;
            ans.assign(1, (char) 0);
            if (op == 'H') {
                authed = token_matches(std::string(req.data() + 1, req.size() - 1), token);
                if (authed) sl::put(ans, (uint32_t) row);
                else ok = false, why = "the stage link's secret does not match";
            } else if (!authed) {
                ok = false;
                why = "the stage link needs its hello first";
            } else if (op == 'Z') {
                m.reset();
            } else if (op == 'N') {
                m.pin();
            } else if (op == 'K') {
                int64_t slot = 0;
                ok = sl::get(req, at, slot);
                if (ok) m.checkpoint((int) slot);
            } else if (op == 'B') {
                int64_t limit = 0;
                ok = sl::get(req, at, limit);
                if (ok) sl::put(ans, (uint8_t) (m.restore(limit) ? 1 : 0));
            } else if (op == 'T') {
                int64_t both = 0;
                ok = sl::get(req, at, both);
                if (ok) m.stage_rollback((int) (both >> 32), (int) (both & 0x7fffffff));
            } else if (op == 'W' || op == 'V') {
                int64_t pos0 = 0;
                uint32_t T = 0;
                ok = sl::get(req, at, pos0) && sl::get(req, at, T) && req.size() == at + (size_t) T * 4;
                if (ok && pos0 != m.pos()) {
                    ok = false;
                    why = "the stage holds position " + std::to_string(m.pos()) + ", asked for " + std::to_string(pos0);
                }
                if (ok) {
                    toks.resize(T);
                    if (T > 0) std::memcpy(toks.data(), req.data() + at, (size_t) T * 4);
                    rows.resize((size_t) T * row);
                    ok = m.stage_forward(toks.data(), (int) T, rows.data(), op == 'V', why);
                }
                if (ok) {
                    tensor_from = 1;
                    sl::put_bytes(ans, rows.data(), rows.size() * 4);
                }
            } else {
                ok = false;
                why = std::string("unknown stage op '") + op + "'";
            }
            if (!ok) {
                ans.assign(1, (char) 1);
                if (why.empty()) why = "a malformed stage request";
                sl::put_bytes(ans, why.data(), why.size());
                tensor_from = 0;
            }
            if (!sl::send_msg(fd, ans, tensor_from)) break;
        }
        ::close(fd);
    }
}

/// The later stage's side.
class Prev {
public:
    bool connect(const std::string& addr, const std::string& token, size_t row_floats, std::string& err) {
        row_ = row_floats;
        fd_ = sl::tcp_open(addr, false, err);
        if (fd_ < 0) return false;
        std::vector<char> req{'H'}, ans;
        sl::put_bytes(req, token.data(), token.size());
        if (!call(req, ans, err)) return false;
        size_t at = 1;
        uint32_t row = 0;
        if (!sl::get(ans, at, row) || row != row_) {
            err = "the earlier stage hands on " + std::to_string(row) + " floats a row, this one reads " +
                  std::to_string(row_);
            return false;
        }
        return simple('Z', 0, false, err);
    }
    bool rows(const int* tokens, int T, int64_t pos0, bool spec, float* out, std::string& err) {
        std::vector<char> req{spec ? 'V' : 'W'}, ans;
        sl::put(req, pos0);
        sl::put(req, (uint32_t) T);
        sl::put_bytes(req, tokens, (size_t) T * 4);
        if (!call(req, ans, err)) return false;
        const size_t want = (size_t) T * row_ * 4;
        if (ans.size() != 1 + want) {
            err = "the earlier stage answered " + std::to_string(ans.size() - 1) + " bytes, expected " + std::to_string(want);
            return false;
        }
        std::memcpy(out, ans.data() + 1, want);
        return true;
    }
    /// The state call the serving stage just made. 'B' answers whether it restored.
    bool sync(char op, int64_t arg, std::string& err) {
        return simple(op, arg, op == 'K' || op == 'B' || op == 'T', err);
    }

private:
    bool simple(char op, int64_t arg, bool with_arg, std::string& err) {
        std::vector<char> req{op}, ans;
        if (with_arg) sl::put(req, arg);
        if (!call(req, ans, err)) return false;
        if (op == 'B') return ans.size() >= 2 && ans[1] == 1;
        return true;
    }
    bool call(const std::vector<char>& req, std::vector<char>& ans, std::string& err) {
        if (fd_ < 0 || !sl::send_msg(fd_, req) || !sl::recv_msg(fd_, ans) || ans.empty()) {
            err = "the link to the earlier stage is lost";
            return false;
        }
        if (ans[0] != 0) {
            err = "the earlier stage: " + std::string(ans.data() + 1, ans.size() - 1);
            return false;
        }
        return true;
    }
    int fd_ = -1;
    size_t row_ = 0;
};
}  // namespace strata::ds4::stage
