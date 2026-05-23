"""High-precision async timer for latency measurement."""

import time
from types import TracebackType


class Timer:
    """Async context manager that tracks wall-clock elapsed time.

    Usage::

        async with Timer() as t:
            await some_async_work()
        print(t.elapsed_ms)
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0

    async def __aenter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds between enter and exit."""
        return (self._end - self._start) * 1_000
