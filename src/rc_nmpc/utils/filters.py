from dataclasses import dataclass
import math

@dataclass
class OneEuro:
    min_cutoff: float = 1.0
    beta: float = 0.0
    d_cutoff: float = 1.0
    _x_prev: float = 0.0
    _dx_prev: float = 0.0
    _t_prev: float = 0.0
    _initialized: bool = False

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t):
        if not self._initialized:
            self._initialized = True
            self._x_prev = x; self._t_prev = t
            return x
        dt = max(1e-6, t - self._t_prev)
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t
        return x_hat
