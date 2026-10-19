"""Run a slow check in a separate process with a hard time limit.

MeshLab's C++ filters cannot be interrupted from Python, so anything that might
run away (for example a self-intersection count on long, thin triangles) runs in
a child process that is killed when its time budget runs out.
"""

from __future__ import annotations

import multiprocessing
import queue
from collections.abc import Callable
from typing import Any


def _child(fn: Callable, args: tuple, results) -> None:
    try:
        results.put(("ok", fn(*args)))
    except BaseException as exc:  # reported to the parent, which decides what to do
        results.put(("error", f"{exc.__class__.__name__}: {exc}"))


def run_with_budget(fn: Callable, args: tuple, seconds: float) -> tuple[str, Any]:
    """Return ("ok", value), ("timeout", None), or ("error", message)."""
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_child, args=(fn, args, results))
    proc.start()
    try:
        status, value = results.get(timeout=seconds)
    except queue.Empty:
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
        return "timeout", None
    proc.join(5)
    return status, value
