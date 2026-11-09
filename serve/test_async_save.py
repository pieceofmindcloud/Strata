"""serve/test_async_save.py - a layer split's SAVE answered SAVING and written later (StrataEngine), no GPU needed.

    python -m unittest serve.test_async_save -v

The engine is a fake: its stdin is recorded, and a thread answers each SAVE on the stdout the server reads.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server as S  # noqa: E402
from serve.server import StrataEngine, SessionRefused  # noqa: E402


class FakeStdin:
    def __init__(self, on_line):
        self.on_line = on_line

    def write(self, text):
        self.on_line(text.rstrip("\n"))

    def flush(self):
        pass


def fake_engine(answers):
    """answers(command) -> the stdout lines the engine prints for it, each given after a delay (seconds)."""
    e = object.__new__(StrataEngine)
    e.batch = 2
    e.slot_busy = [False, False]
    e.slot_held = [[1, 2, 3], []]
    e.slot_used = [0.0, 0.0]
    e.slot_cv = threading.Condition()
    e.silence_s = 0
    e.save_q = queue.Queue()
    e.lines = queue.Queue()
    e.save_pending = None
    e.commands = []

    def on_line(command):
        e.commands.append(command)
        for delay, line in answers(command):
            def send(line=line, delay=delay):
                time.sleep(delay)
                e.save_q.put(line + "\n")
            threading.Thread(target=send, daemon=True).start()

    e.proc = types.SimpleNamespace(stdin=FakeStdin(on_line), poll=lambda: None)
    e.exit_code = lambda: None
    return e


def saving_then(saved_line, delay=0.2):
    def answers(command):
        if command.startswith("SAVE @"):
            return [(0.0, "SAVING 18236 12.5"), (delay, saved_line)]
        return []
    return answers


class AsyncSave(unittest.TestCase):
    def test_saving_frees_the_slot_before_the_file_is_written(self):
        e = fake_engine(saving_then("SAVED 18236 396939976 850.0", delay=0.5))
        started = time.monotonic()
        result = e.session_file("save", "/s/a.strsess", slot=0)
        self.assertLess(time.monotonic() - started, 0.4)      # the answer came at SAVING, not at SAVED
        self.assertTrue(result["pending"])
        self.assertEqual(result["tokens"], 18236)
        self.assertFalse(e.slot_busy[0])                       # the slot is free while the file is written
        self.assertEqual(e.commands, ["SAVE @0 /s/a.strsess"])

    def test_the_waiter_gets_the_saved_numbers(self):
        e = fake_engine(saving_then("SAVED 18236 396939976 850.0"))
        ticket = e.session_file("save", "/s/a.strsess", slot=0)["pending"]
        done = e.await_session_file(ticket)
        self.assertEqual((done["tokens"], done["bytes"], done["ms"]), (18236, 396939976, 850.0))
        self.assertIsNone(e.save_pending)

    def test_a_refused_write_raises_to_its_caller_only(self):
        e = fake_engine(saving_then("SERR storage 0 the disk is full"))
        ticket = e.session_file("save", "/s/a.strsess", slot=0)["pending"]
        with self.assertRaises(SessionRefused):
            e.await_session_file(ticket)
        self.assertIsNone(e.save_pending)

    def test_the_next_session_command_reads_the_pending_saved_first(self):
        e = fake_engine(saving_then("SAVED 18236 396939976 850.0"))
        e.session_file("save", "/s/a.strsess", slot=0)
        # a restore is sent only after the earlier SAVED was read: its answer must not be taken as the restore's
        e.commands.clear()
        e.proc.stdin.on_line = lambda c: (e.commands.append(c),
                                          e.save_q.put("RESTORED 18236 396939976 40.0\n"))
        result = e.session_file("restore", "/s/a.strsess", slot=1, ids=[1, 2])
        self.assertEqual(result["tokens"], 18236)
        self.assertIsNone(e.save_pending)

    def test_a_synchronous_saved_still_answers_the_save(self):
        def answers(command):
            return [(0.0, "SAVED 3 100 5.0")] if command.startswith("SAVE @") else []
        e = fake_engine(answers)
        result = e.session_file("save", "/s/b.strsess", slot=0)
        self.assertEqual(result["bytes"], 100)
        self.assertNotIn("pending", result)

    def test_session_replies_never_reach_the_generation_lines(self):
        e = object.__new__(StrataEngine)
        e.save_q = queue.Queue()
        e.lines = queue.Queue()
        e.slot_q = []
        e.slot_busy = []
        e.rid_q = {}
        e.multi = 0
        e.ended = False
        e.proc = types.SimpleNamespace(stdout=iter([
            "SAVING 5 1.0\n", "SAVED 5 10 2.0\n", "SERR storage 0 full\n", "RESTORED 5 10 1.0\n",
            "T 42\n", "DONE stop\n",
        ]))
        e._pump()
        generation = []
        while not e.lines.empty():
            generation.append(e.lines.get())
        self.assertEqual(generation[:2], ["T 42\n", "DONE stop\n"])
        replies = [e.save_q.get() for _ in range(4)]
        self.assertEqual([r.split()[0] for r in replies], ["SAVING", "SAVED", "SERR", "RESTORED"])


if __name__ == "__main__":
    unittest.main()
