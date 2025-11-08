import time
def now_ns(): return time.monotonic_ns()
class Rate:
    def __init__(self, hz: float):
        self.period = 1.0 / hz
        self._last = time.perf_counter()
    def sleep(self):
        now = time.perf_counter()
        rem = self.period - (now - self._last)
        if rem > 0: time.sleep(rem)
        self._last = time.perf_counter()
