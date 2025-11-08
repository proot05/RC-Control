from dataclasses import dataclass
import time

# -------------------- Servo --------------------

@dataclass
class ServoConfig:
    min_us: int = 1000
    max_us: int = 2000
    zero_us: int = 1500
    delta_max_rad: float = 0.6
    rate_max_rad_s: float = 3.0

class SteeringMapper:
    """
    Rate-limited steering: maps desired wheel angle (rad) to servo µs.
    Preserves your original API & behavior.
    """
    def __init__(self, cfg: ServoConfig):
        self.cfg = cfg
        self._last_delta = 0.0
        self._last_t = time.monotonic()

    def delta_to_us(self, delta_rad: float) -> int:
        # clamp to mechanical range
        d = max(-self.cfg.delta_max_rad, min(self.cfg.delta_max_rad, float(delta_rad)))

        # rate limit
        t = time.monotonic()
        dt = max(1e-3, t - self._last_t)
        max_step = self.cfg.rate_max_rad_s * dt
        d = max(self._last_delta - max_step, min(self._last_delta + max_step, d))

        # store state
        self._last_delta, self._last_t = d, t

        # linear map to [min_us, max_us] centered on zero_us
        span = self.cfg.max_us - self.cfg.min_us
        frac = 0.5 + 0.5 * (d / self.cfg.delta_max_rad)  # [-dmax,+dmax] -> [0,1]
        us = int(self.cfg.min_us + span * frac)
        return max(self.cfg.min_us, min(self.cfg.max_us, us))


# -------------------- ESC  (bi-direction + brake→reverse) --------------------

@dataclass
class EscConfig:
    # Base PWM calibration
    min_us: int = 1000
    max_us: int = 2000
    zero_us: int = 1500
    deadband_us: int = 15

    # Legacy single-direction mapping (kept for compatibility)
    # Used if bidirectional == False
    accel_to_us_gain: float = 120.0

    # Bidirectional behavior
    bidirectional: bool = True
    forward_scale: float = 1.0     # scale forward mapping (sensitivity)
    reverse_scale: float = 1.0     # scale reverse mapping (sensitivity)

    # Common RC ESC policy: must apply brake before reverse engages
    require_brake_before_reverse: bool = True
    brake_pulse_us: int = 1300     # µs to send during brake phase (below neutral)
    brake_time_ms: int = 250       # how long to hold brake pulse before reverse


class EscMapper:
    """
    Maps accel command to ESC PWM (µs).

    API:
      - ax_to_us(ax_cmd: float) -> int
        * If cfg.bidirectional == False:
            uses legacy linear map around zero_us via accel_to_us_gain
        * If cfg.bidirectional == True:
            ax_cmd ∈ [-1,1]: negative = brake/reverse, positive = forward
            applies optional brake→reverse timing policy

      - arm_sequence(writer_func): sends min -> max -> zero (as you had)
    """
    def __init__(self, cfg: EscConfig):
        self.cfg = cfg
        # reverse/brake state for brake-before-reverse logic
        self._braked_once = False
        self._brake_until_ns = 0

    def _linear_one_direction(self, ax_cmd: float) -> int:
        """Legacy single-direction mapping (your original behavior)."""
        us = int(self.cfg.zero_us + float(ax_cmd) * self.cfg.accel_to_us_gain)
        if abs(us - self.cfg.zero_us) < self.cfg.deadband_us:
            us = self.cfg.zero_us
        return max(self.cfg.min_us, min(self.cfg.max_us, us))

    def ax_to_us(self, ax_cmd: float) -> int:
        cfg = self.cfg
        zero = cfg.zero_us

        # If not bidirectional, use legacy linear mapping around zero
        if not cfg.bidirectional:
            return self._linear_one_direction(ax_cmd)

        # Bidirectional: clamp to [-1, 1]
        cmd = max(-1.0, min(1.0, float(ax_cmd)))
        now_ns = time.monotonic_ns()

        # ----- Forward -----
        if cmd > 0.0:
            # clear brake latch when re-entering forward
            self._braked_once = False
            us = zero + cmd * (cfg.max_us - zero) * cfg.forward_scale

        # ----- Reverse / Brake -----
        elif cmd < 0.0:
            if cfg.require_brake_before_reverse:
                # if we haven't braked yet, start brake window
                if not self._braked_once:
                    self._braked_once = True
                    self._brake_until_ns = now_ns + int(cfg.brake_time_ms * 1e6)
                    # brake pulse safely below neutral (never exceed [min, zero))
                    us = max(cfg.min_us, min(cfg.brake_pulse_us, zero - 1))
                    # deadband does not apply during explicit brake pulse
                    return us

                # still braking?
                if now_ns < self._brake_until_ns:
                    us = max(cfg.min_us, min(cfg.brake_pulse_us, zero - 1))
                    return us

                # brake complete -> allow reverse proportionally
                us = zero + cmd * (zero - cfg.min_us) * cfg.reverse_scale
            else:
                # no brake requirement -> direct reverse
                us = zero + cmd * (zero - cfg.min_us) * cfg.reverse_scale

        # ----- Neutral -----
        else:
            self._braked_once = False
            us = zero

        # Small neutral deadband to avoid twitch
        if abs(us - zero) < cfg.deadband_us:
            us = zero

        # Clamp to valid range
        return max(cfg.min_us, min(cfg.max_us, int(us)))

    def arm_sequence(self, writer_func):
        """
        arming sequence: min -> max -> zero (each 1s).
        Keep if  ESC expects this; otherwise replace with
        a neutral-hold here.
        """
        print("If you don't have the ESC off start over...")
        writer_func(self.cfg.zero_us)
        print("Neutral signal set, you have 12 s to turn on the ESC and hear some noises?...")
        time.sleep(12.0)
        #print("Hopefully you heard 2 beeps and a falling tone after turning on the ESC?...")
        #writer_func(self.cfg.min_us)
        #time.sleep(12.0)
        print("Hopefully you heard a special tone?...")
        writer_func(0)
        time.sleep(2)
        print("Testing and arming ESC...")
        print("Reverse")
        writer_func(self.cfg.min_us);time.sleep(1.0)
        print("Forward")
        writer_func(self.cfg.max_us);time.sleep(1.0)
        print("Neutral")
        writer_func(self.cfg.zero_us); time.sleep(1.0)
