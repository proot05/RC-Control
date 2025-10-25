from dataclasses import dataclass
import time

@dataclass
class ServoConfig:
    min_us: int = 1000
    max_us: int = 2000
    zero_us: int = 1500
    delta_max_rad: float = 0.6
    rate_max_rad_s: float = 3.0

@dataclass
class EscConfig:
    min_us: int = 1000
    max_us: int = 2000
    zero_us: int = 1500
    accel_to_us_gain: float = 120.0
    deadband_us: int = 15

class SteeringMapper:
    def __init__(self, cfg: ServoConfig):
        self.cfg = cfg
        self._last_delta = 0.0
        self._last_t = time.monotonic()
    def delta_to_us(self, delta_rad: float) -> int:
        d = max(-self.cfg.delta_max_rad, min(self.cfg.delta_max_rad, delta_rad))
        t = time.monotonic(); dt = max(1e-3, t - self._last_t)
        max_step = self.cfg.rate_max_rad_s * dt
        d = max(self._last_delta - max_step, min(self._last_delta + max_step, d))
        self._last_delta, self._last_t = d, t
        span = self.cfg.max_us - self.cfg.min_us
        frac = 0.5 + 0.5 * (d / self.cfg.delta_max_rad)
        return int(self.cfg.min_us + span * frac)

class EscMapper:
    def __init__(self, cfg: EscConfig): self.cfg = cfg
    def ax_to_us(self, ax_cmd: float) -> int:
        us = int(self.cfg.zero_us + ax_cmd * self.cfg.accel_to_us_gain)
        if abs(us - self.cfg.zero_us) < self.cfg.deadband_us:
            us = self.cfg.zero_us
        return max(self.cfg.min_us, min(self.cfg.max_us, us))
    def arm_sequence(self, writer_func):
        writer_func(self.cfg.min_us); time.sleep(1.0)
        writer_func(self.cfg.max_us); time.sleep(1.0)
        writer_func(self.cfg.zero_us); time.sleep(1.0)
