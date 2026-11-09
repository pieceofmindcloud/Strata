"""pack_gguf --layers: the pack of a range of the whole model, renumbered like a slice's.  The numbers are
DeepSeek-V4.1-Flash's: KV sources 2, 8, 14, 20, indexer sources up to 36, Engram at 1 and 14, candidate source 20."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "dsv41"))
import pack_gguf as PG  # noqa: E402

A = "deepseek41"


def model_kv() -> dict:
    ngram, heads = 4, 8
    cols = (ngram - 1) * heads
    return {
        "general.architecture": A,
        A + ".block_count": 40,
        A + ".attention.compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
        A + ".swiglu_clamp_exp": [10.0] * 40,
        A + ".swiglu_clamp_shexp": [10.0] * 40,
        A + ".attention.kv_source_layers": [2, 8, 14, 20],
        A + ".attention.indexer.source_layers": [2, 8, 14, 20, 24, 28, 32, 36],
        A + ".attention.indexer.candidate_source_layer": 20,
        A + ".engram.layers": [1, 14],
        A + ".engram.max_ngram_size": ngram,
        A + ".engram.head_count": heads,
        A + ".engram.multipliers": list(range(2 * ngram)),
        A + ".engram.primes": list(range(100, 100 + 2 * cols)),
        A + ".engram.offsets": list(range(1000, 1000 + 2 * cols)),
        A + ".engram.num_embeddings": [7, 9],
    }


def tensors() -> list:
    t = [("embed.weight", "BF16", 0, 0, 1, [1]), ("head.weight", "BF16", 0, 1, 1, [1])]
    return t + [("layers.%d.attn.wq_a.weight" % il, "BF16", 0, 10 + il, 1, [1]) for il in range(40)]


class CutLayers(unittest.TestCase):
    def test_a_middle_range_with_one_engram_layer(self):
        kv, ts = model_kv(), tensors()
        PG.cut_layers(kv, ts, 14, 19)
        self.assertEqual(kv[A + ".block_count"], 6)
        self.assertEqual(kv[A + ".attention.compress_ratios"], [2] * 6)
        self.assertEqual(kv[A + ".swiglu_clamp_exp"], [10.0] * 6)
        self.assertEqual(kv[A + ".attention.kv_source_layers"], [0])
        self.assertEqual(kv[A + ".attention.indexer.source_layers"], [0])
        self.assertNotIn(A + ".attention.indexer.candidate_source_layer", kv, "the source is past the range")
        self.assertEqual(kv[A + ".engram.layers"], [0])
        self.assertEqual(kv[A + ".engram.multipliers"], [4, 5, 6, 7], "layer 14's own, the second table's")
        self.assertEqual(kv[A + ".engram.primes"], list(range(124, 148)))
        self.assertEqual(kv[A + ".engram.offsets"], list(range(1024, 1048)))
        self.assertEqual(kv[A + ".engram.num_embeddings"], [9])
        self.assertEqual((kv["pom.start_layer"], kv["pom.end_layer"], kv["pom.original_block_count"]), (14, 19, 40))
        names = [t[0] for t in ts]
        self.assertEqual(names[:2], ["embed.weight", "head.weight"], "every slice keeps the embedding and the head")
        self.assertEqual(names[2:], ["layers.%d.attn.wq_a.weight" % il for il in range(6)])
        self.assertEqual(ts[2][3], 10 + 14, "renamed, the same bytes")

    def test_the_last_range_keeps_the_candidate_source_rebased(self):
        kv, ts = model_kv(), tensors()
        PG.cut_layers(kv, ts, 20, 39)
        self.assertEqual(kv[A + ".attention.indexer.candidate_source_layer"], 0)
        self.assertEqual(kv[A + ".attention.indexer.source_layers"], [0, 4, 8, 12, 16])
        self.assertEqual(kv[A + ".engram.layers"], [])
        self.assertEqual(kv[A + ".attention.compress_ratios"], [1] * 20, "the drafter adds its own ratios")

    def test_a_range_outside_the_model_is_refused(self):
        with self.assertRaises(SystemExit):
            PG.cut_layers(model_kv(), tensors(), 30, 40)


if __name__ == "__main__":
    unittest.main()
