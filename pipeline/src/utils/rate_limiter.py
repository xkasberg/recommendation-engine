import threading
import time
from collections import deque


class RateLimiter:
    """Thread-safe token bucket that enforces `max_calls` per `period` seconds."""
    def __init__(self, max_calls: int, period: float = 60.0) -> None:
        self.max_calls = max_calls
        self.period = period
        self._calls = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """
        Block the caller until performing another request keeps us within the
        configured rate. Safe to call from multiple threads.
        """
        while True:
            with self._lock:
                now = time.time()

                # Drop calls outside the rolling window
                while self._calls and (now - self._calls[0]) > self.period:
                    self._calls.popleft()

                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return

                earliest = self._calls[0]
                sleep_for = max(0.0, self.period - (now - earliest))

            if sleep_for > 0:
                time.sleep(sleep_for)
