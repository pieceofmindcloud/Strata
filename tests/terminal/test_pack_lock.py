"""Two writers of one pack folder: the second waits for the first and is told the pack was written."""
import multiprocessing
import pathlib
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
import pack_lock  # noqa: E402


def _hold(folder: str, ready, seconds: float) -> None:
    pack_lock.acquire(folder)
    ready.set()
    time.sleep(seconds)


class PackLockTest(unittest.TestCase):
    def test_a_free_pack_is_locked_without_waiting(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(pack_lock.acquire(pathlib.Path(folder) / "pack"))

    def test_the_second_writer_waits_for_the_first(self):
        with tempfile.TemporaryDirectory() as folder:
            pack = str(pathlib.Path(folder) / "pack")
            ready = multiprocessing.Event()
            first = multiprocessing.Process(target=_hold, args=(pack, ready, 1.0))
            first.start()
            self.assertTrue(ready.wait(10))
            started = time.monotonic()
            self.assertTrue(pack_lock.acquire(pack))
            self.assertGreater(time.monotonic() - started, 0.5)
            first.join()


if __name__ == "__main__":
    unittest.main()
