// The DeepSeek-V4.1 stage link's micro-pipeline without a model or a GPU (ds4_stage.hpp, Prev::stream_*): a prompt's
// chunks asked for ahead against a fake earlier stage that answers 'W' in order and, as the real one, only at the
// position it holds.
#include "../../src/dsv41/ds4_stage.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace sl = strata::stage;
namespace st = strata::ds4::stage;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

constexpr uint32_t kRow = 64;
float cell(int tok, uint32_t col) { return (float) (tok % 7) + (float) col / 256.0f; }

// The fake earlier stage: 'H' (its row width), 'Z' (back to position 0), 'K' (a checkpoint, nothing kept), and
// 'W' chunks answered in order after a short pause each (its compute) - refused when they do not start where it is.
void serve(int lfd, std::atomic<int>& refused) {
    const int cfd = ::accept(lfd, nullptr, nullptr);
    int64_t pos = 0;
    std::vector<char> req;
    while (sl::recv_msg(cfd, req)) {
        std::vector<char> ans{0};
        size_t at = 1, tensor_from = 0;
        if (req[0] == 'H') {
            sl::put(ans, kRow);
        } else if (req[0] == 'Z') {
            pos = 0;
        } else if (req[0] == 'W') {
            int64_t pos0 = 0;
            uint32_t T = 0;
            sl::get(req, at, pos0);
            sl::get(req, at, T);
            if (pos0 != pos) {
                ++refused;
                ans.assign(1, (char) 1);
                const std::string why = "the stage holds position " + std::to_string(pos);
                sl::put_bytes(ans, why.data(), why.size());
            } else {
                std::vector<float> rows((size_t) T * kRow);
                for (uint32_t t = 0; t < T; ++t) {
                    int32_t tok = 0;
                    std::memcpy(&tok, req.data() + at + 4 * t, 4);
                    for (uint32_t c = 0; c < kRow; ++c) rows[(size_t) t * kRow + c] = cell(tok, c);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                pos += T;
                tensor_from = 1;
                sl::put_bytes(ans, rows.data(), rows.size() * sizeof(float));
            }
        }
        if (!sl::send_msg(cfd, ans, tensor_from)) break;
    }
    ::close(cfd);
}

void stream_against_a_fake_stage() {
    std::string err;
    int port = 0;
    const int lfd = sl::tcp_open("127.0.0.1:0", true, err, &port);
    check(lfd >= 0, "the fake stage listens");
    std::atomic<int> refused{0};
    std::thread server([&] { serve(lfd, refused); });
    {
        st::Prev prev;
        check(prev.connect("127.0.0.1:" + std::to_string(port), "", kRow, err), "the later stage connects");
        std::vector<int> toks(40);
        for (size_t i = 0; i < toks.size(); ++i) toks[i] = (int) i * 3;
        std::vector<st::Prev::Piece> pieces;
        for (int64_t a = 0; a < 40; a += 8) pieces.push_back({a, 8});

        // a whole prompt read with up to 4 chunks ahead: in order, several in flight
        check(prev.stream_begin(toks.data(), pieces, 4, err), "the stream starts");
        std::vector<float> rows(8 * kRow);
        bool exact = true;
        for (int64_t c = 0; c < 40; c += 8) {
            check(prev.stream_take(8, rows.data(), err), "a chunk is taken");
            std::this_thread::sleep_for(std::chrono::milliseconds(15));   // this stage's compute
            for (int64_t t = 0; t < 8; ++t)
                for (uint32_t col = 0; col < kRow; ++col)
                    if (rows[(size_t) t * kRow + col] != cell(toks[(size_t) (c + t)], col)) exact = false;
        }
        check(exact, "the rows come back in the order of the chunks");
        check(prev.stream_end() >= 2, "several chunks were in flight at once");
        check(refused == 0, "every chunk started where the earlier stage was");

        // a state call after the stream: the link answers it, not a leftover chunk
        check(prev.sync('K', 1, err), "a checkpoint after the stream");

        // a run cut short (a Stop): its answers are read, the earlier stage is ahead, the server's reset brings both
        // back to position 0 and the next stream reads from there
        check(prev.sync('Z', 0, err), "a reset");
        check(prev.stream_begin(toks.data(), pieces, 3, err), "a second stream starts");
        check(prev.stream_take(8, rows.data(), err), "its first chunk is taken");
        prev.stream_end();
        check(prev.sync('Z', 0, err), "the reset after the early end");
        check(prev.stream_begin(toks.data(), pieces, 2, err), "a third stream starts after the early end");
        check(prev.stream_take(8, rows.data(), err) && rows[0] == cell(toks[0], 0),
              "its first chunk is its own, not a leftover");
        prev.stream_end();

        // a chunk the earlier stage refuses (it holds another position): the take fails with its reason, and the link
        // stays usable
        check(prev.sync('Z', 0, err), "a reset before the refused stream");
        std::vector<st::Prev::Piece> wrong{{8, 8}, {16, 8}};
        check(prev.stream_begin(toks.data() + 8, wrong, 2, err), "a stream at the wrong position starts");
        check(!prev.stream_take(8, rows.data(), err) && err.find("holds position 0") != std::string::npos,
              "its first chunk fails with the earlier stage's reason");
        prev.stream_end();
        check(prev.sync('Z', 0, err), "the link still answers after the refusal");
        check(prev.rows(toks.data(), 8, 0, false, rows.data(), err) && rows[0] == cell(toks[0], 0),
              "and a plain chunk after it is in step");
    }
    server.join();
    ::close(lfd);
}
}  // namespace

int main() {
    stream_against_a_fake_stage();
    std::printf("ds4_stage_stream_test: %d checks passed\n", checks);
    return 0;
}
