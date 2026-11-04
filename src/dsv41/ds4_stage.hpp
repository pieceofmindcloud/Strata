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
//
// THE MICRO-PIPELINE of a prompt read (Prev::stream_begin, as the other engine's PrevStage): a prompt's chunks are
// asked for ahead - their 'W' requests sent before the answers to the earlier ones are read - up to
// STRATA_STAGE_INFLIGHT chunks (8 by default, 1 = one at a time) and STRATA_STAGE_INFLIGHT_MIB of rows (512 by default)
// beyond the ones this stage has taken. The earlier stage answers in order and finds the next request already in its
// socket, so it reads chunk k + 1 while its answer to chunk k crosses the network and this stage reads chunk k. The
// earlier stage serves exactly as before. A run that ends early (a Stop, an error) reads the answers already asked
// for (the connection stays in step); the earlier stage may then hold positions past this one's, which the server's
// restore ('B') or reset ('Z') after such a run brings back in step, as it does for a run cut short without a stream.
#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
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

/// How many chunks of `chunk_rows` rows a prompt read keeps asked for ahead: STRATA_STAGE_INFLIGHT (8 by default,
/// 1 = one at a time), and no more than STRATA_STAGE_INFLIGHT_MIB of rows (512 by default) - the same knobs, and the
/// same defaults, as the other engine's micro-pipeline.
inline int stream_depth(size_t row_floats, int chunk_rows) {
    const char* v = std::getenv("STRATA_STAGE_INFLIGHT");
    const char* mib = std::getenv("STRATA_STAGE_INFLIGHT_MIB");
    const int64_t d = v != nullptr && std::atoi(v) > 0 ? std::atoi(v) : 8;
    const int64_t budget = (int64_t) (mib != nullptr && std::atoll(mib) > 0 ? std::atoll(mib) : 512) << 20;
    const int64_t per = (int64_t) std::max(1, chunk_rows) * (int64_t) row_floats * (int64_t) sizeof(float);
    return (int) std::max<int64_t>(1, std::min<int64_t>(d, budget / std::max<int64_t>(per, 1)));
}

/// The later stage's side.
class Prev {
public:
    ~Prev() {
        stream_end();
        if (fd_ >= 0) ::close(fd_);
    }

    /// A chunk of a prompt read through the earlier stage: one 'W' request.
    struct Piece {
        int64_t pos0 = 0;
        int T = 0;
    };
    /// THE MICRO-PIPELINE (see the top of this file): a thread sends the pieces' 'W' requests - `tokens` holds the
    /// prompt from the first piece's position, copied here - and keeps up to `depth` of them ahead of the ones
    /// stream_take has taken. Nothing else may use the link until stream_end.
    bool stream_begin(const int* tokens, std::vector<Piece> pieces, int depth, std::string& err) {
        stream_end();
        if (fd_ < 0) {
            err = "the link to the earlier stage is lost";
            return false;
        }
        st_pieces_ = std::move(pieces);
        const int64_t base = st_pieces_.empty() ? 0 : st_pieces_.front().pos0;
        const int64_t end = st_pieces_.empty() ? 0 : st_pieces_.back().pos0 + st_pieces_.back().T;
        st_tokens_.assign(tokens, tokens + (end - base));
        st_base_ = base;
        st_depth_ = std::max(1, depth);
        st_sent_ = st_taken_ = 0;
        st_ready_.clear();
        st_err_.clear();
        st_stop_ = st_failed_ = false;
        st_depth_max_ = 0;
        st_thread_ = std::thread([this] { stream_loop(); });
        return true;
    }
    /// The rows of the stream's next piece (T rows, as stream_begin was given), in order.
    bool stream_take(int T, float* out, std::string& err) {
        std::vector<float> ans;
        {
            std::unique_lock<std::mutex> lk(st_mu_);
            st_cv_.wait(lk, [&] { return !st_ready_.empty() || st_failed_; });
            if (st_ready_.empty()) {
                err = st_err_;
                return false;
            }
            if (st_pieces_[st_taken_].T != T) {
                err = "the stage stream's piece has " + std::to_string(st_pieces_[st_taken_].T) + " rows, the chunk " +
                      std::to_string(T);
                return false;
            }
            ans = std::move(st_ready_.front());
            st_ready_.pop_front();
            ++st_taken_;
        }
        st_cv_.notify_all();
        std::memcpy(out, ans.data(), ans.size() * sizeof(float));
        return true;
    }
    /// Ends the stream: no new piece is asked for and the answers already asked for are read (the connection stays
    /// in step). The most pieces that were in flight at once (0: no stream).
    int stream_end() {
        if (st_thread_.joinable()) {
            {
                std::lock_guard<std::mutex> lk(st_mu_);
                st_stop_ = true;
            }
            st_cv_.notify_all();
            st_thread_.join();
        }
        st_ready_.clear();
        st_pieces_.clear();
        st_tokens_.clear();
        const int d = st_depth_max_;
        st_depth_max_ = 0;
        return d;
    }

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
        stream_end();   // (a stream's answers are read first: the link answers in order)
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

    void stream_fail(const std::string& e) {
        {
            std::lock_guard<std::mutex> lk(st_mu_);
            if (!st_failed_) st_err_ = e;
            st_failed_ = true;
        }
        st_cv_.notify_all();
    }
    void lost() {
        stream_fail("the link to the earlier stage is lost");
        ::close(fd_);
        fd_ = -1;
    }
    // after a failed piece: the answers to the pieces already sent still come, read and dropped
    void drain(size_t received) {
        std::vector<char> ans;
        for (; received < st_sent_; ++received)
            if (!sl::recv_msg(fd_, ans)) {
                lost();
                return;
            }
    }
    // The stream's thread: sends while fewer than `depth` pieces are ahead of the taken ones (the answers received and
    // not taken count, so their memory stays bounded), else reads the oldest answer. After stream_end it sends nothing
    // more and reads what it has sent.
    void stream_loop() {
        size_t received = 0;
        for (;;) {
            bool send_one = false;
            {
                std::unique_lock<std::mutex> lk(st_mu_);
                st_cv_.wait(lk, [&] {
                    return st_stop_ || st_failed_ || received < st_sent_ ||
                           (st_sent_ < st_pieces_.size() && st_sent_ - st_taken_ < (size_t) st_depth_);
                });
                if (st_failed_ || (st_stop_ && received == st_sent_)) return;
                send_one = !st_stop_ && st_sent_ < st_pieces_.size() && st_sent_ - st_taken_ < (size_t) st_depth_;
                if (!send_one && received == st_sent_) continue;
            }
            if (send_one) {
                const Piece& p = st_pieces_[st_sent_];
                std::vector<char> req{'W'};
                sl::put(req, p.pos0);
                sl::put(req, (uint32_t) p.T);
                sl::put_bytes(req, st_tokens_.data() + (p.pos0 - st_base_), (size_t) p.T * 4);
                if (!sl::send_msg(fd_, req)) {
                    lost();
                    return;
                }
                std::lock_guard<std::mutex> lk(st_mu_);
                ++st_sent_;
                st_depth_max_ = std::max(st_depth_max_, (int) (st_sent_ - received));
                // keep sending while the window has room: the answers wait in the socket
                if (st_sent_ < st_pieces_.size() && st_sent_ - st_taken_ < (size_t) st_depth_ && !st_stop_) continue;
            }
            std::vector<char> ans;
            if (!sl::recv_msg(fd_, ans)) {
                lost();
                return;
            }
            const Piece& p = st_pieces_[received++];
            if (ans.empty() || ans[0] != 0) {
                stream_fail("the earlier stage: " +
                            (ans.size() > 1 ? std::string(ans.data() + 1, ans.size() - 1) : std::string("error")));
                drain(received);
                return;
            }
            const size_t want = (size_t) p.T * row_;
            if (ans.size() != 1 + want * sizeof(float)) {
                stream_fail("the earlier stage answered " + std::to_string(ans.size() - 1) + " bytes, expected " +
                            std::to_string(want * sizeof(float)));
                drain(received);
                return;
            }
            std::vector<float> rows(want);
            std::memcpy(rows.data(), ans.data() + 1, want * sizeof(float));
            {
                std::lock_guard<std::mutex> lk(st_mu_);
                if (!st_stop_) st_ready_.push_back(std::move(rows));
            }
            st_cv_.notify_all();
        }
    }

    int fd_ = -1;
    size_t row_ = 0;
    // the stream (stream_begin): its pieces and their tokens, how many were sent and taken, the answers not taken yet
    std::vector<Piece> st_pieces_;
    std::vector<int> st_tokens_;
    int64_t st_base_ = 0;
    int st_depth_ = 1, st_depth_max_ = 0;
    size_t st_sent_ = 0, st_taken_ = 0;
    std::deque<std::vector<float>> st_ready_;
    std::string st_err_;
    bool st_stop_ = false, st_failed_ = false;
    std::mutex st_mu_;
    std::condition_variable st_cv_;
    std::thread st_thread_;
};
}  // namespace strata::ds4::stage
