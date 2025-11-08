from collections import deque
from threading import Lock
from typing import Optional, Deque, TypeVar, Generic

T = TypeVar("T")

class RingBuffer(Generic[T]):
    def __init__(self, capacity: int = 1024):
        self._buf: Deque[T] = deque(maxlen=capacity)
        self._lock = Lock()
    def push(self, item: T) -> None:
        with self._lock:
            self._buf.append(item)
    def latest(self) -> Optional[T]:
        with self._lock:
            return self._buf[-1] if self._buf else None
    def pop_older_than(self, t_ns: int):
        with self._lock:
            while self._buf and getattr(self._buf[0], "t", getattr(self._buf[0], "t_cam", 0)) < t_ns:
                self._buf.popleft()
