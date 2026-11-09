"""serve/test_slot_figures.py - a batch slot's BDONE figures reach its request when the turn stops early, no GPU needed.

    python -m unittest serve.test_slot_figures -v
"""
from __future__ import annotations

import queue
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import StrataEngine  # noqa: E402


class SlotFigures(unittest.TestCase):
    def test_a_slot_stopped_before_its_end_keeps_its_decode_time(self):
        # a reasoning budget's wrap-up BSTOPs the slot: its BDONE carries the leg's decode time and tokens, which the
        # request's figures must hold (they were the admission's single token)
        e = object.__new__(StrataEngine)
        e.slot_q = [queue.Queue()]
        e.last = {"prompt_ms": 3431.0, "decode_ms": 27.0, "generated": 1, "finish": "length"}
        for line in ("BT 0 11", "BT 0 12", "BDONE 0 300 cancel 6380.5"):
            e.slot_q[0].put(line)
        finish, tokens = e._await_bdone(0, [1, 2, 3])
        self.assertEqual(finish, "cancel")
        self.assertEqual(tokens, [1, 2, 3, 11, 12])
        self.assertEqual(e.last["decode_ms"], 6407.5)  # upstream totals admission (27) and slot (6380.5)
        self.assertEqual(e.last["generated"], 300)
        self.assertEqual(e.last["prompt_ms"], 3431.0)


if __name__ == "__main__":
    unittest.main()
