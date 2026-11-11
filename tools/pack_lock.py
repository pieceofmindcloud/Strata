"""One writer per pack folder at a time.

Two engines on one machine (two POM nodes sharing a pack folder) can ask for
the same pack together; the pack tools write fixed temporary names
(`dense.bin.tmp`, `index.txt.tmp`, ...) and rename them, so two writers renamed
each other's files and one failed with FileNotFoundError.  `acquire(out)` takes
an exclusive lock on `<out>/.pack.lock` for the life of the process: the first
writer packs, a second one waits for it and is told so - its pack was just
written, so it has nothing left to write.
"""
from __future__ import annotations

import os
import pathlib
import sys

_held = []   # the lock files, kept open (and locked) until the process exits


def acquire(out: os.PathLike | str) -> bool:
    """Lock the pack folder `out` (created if missing).  Returns True when another
    process held it and has finished: that process wrote the pack."""
    folder = pathlib.Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    handle = open(folder / ".pack.lock", "a+b")
    waited = False
    try:
        import fcntl
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another process is writing this pack (%s): waiting for it" % folder, flush=True)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            waited = True
    except ImportError:   # Windows
        import msvcrt
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print("another process is writing this pack (%s): waiting for it" % folder, flush=True)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)   # retries for 10 s, then raises
                    break
                except OSError:
                    continue
            waited = True
    _held.append(handle)
    return waited


if __name__ == "__main__":   # a manual check: `python pack_lock.py DIR` holds DIR's lock until Enter
    print("waited" if acquire(sys.argv[1]) else "locked")
    input()
