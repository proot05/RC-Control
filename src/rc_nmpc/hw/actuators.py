from dataclasses import dataclass
from typing import Optional, Tuple

from .mapping import ServoConfig, EscConfig, SteeringMapper, EscMapper
from .backends.pigpio_pwm import PigpioPwm, PigpioPins


# -------------------- Backend wrapper (kept as you had it) --------------------

@dataclass
class PigpioBackend:
    servo_gpio: int
    esc_gpio: int

    def __post_init__(self):
        # PigpioPwm handles connecting to pigpio and writing pulses
        self._impl = PigpioPwm(PigpioPins(self.servo_gpio, self.esc_gpio))

    def write_servo(self, us: int) -> None:
        self._impl.write_servo(int(us))

    def write_esc(self, us: int) -> None:
        self._impl.write_esc(int(us))

    def stop(self) -> None:
        self._impl.stop()


# -------------------- High-level actuators --------------------

class Actuators:
    """
    High-level actuator interface:
      - Steering via SteeringMapper (rate-limited -> µs)
      - ESC via EscMapper (supports bidirectional + brake→reverse logic)
      - Center/arm/stop helpers
    """
    def __init__(
        self,
        backend: PigpioBackend,
        servo_cfg: Optional[ServoConfig] = None,
        esc_cfg: Optional[EscConfig] = None,
    ):
        self.backend = backend
        self.servo_cfg = servo_cfg or ServoConfig()
        self.esc_cfg = esc_cfg or EscConfig()

        self.servo_map = SteeringMapper(self.servo_cfg)
        self.esc_map = EscMapper(self.esc_cfg)

        # cache last outputs (useful for diagnostics)
        self._last_servo_us: Optional[int] = None
        self._last_esc_us: Optional[int] = None

        # initialize to neutral
        self.center()

    # ---- Public control API ----

    def arm_esc(self) -> None:
        """
        Run the ESC arming sequence defined by EscMapper.
        If your ESC expects a different sequence, edit EscMapper.arm_sequence().
        """
        self.esc_map.arm_sequence(self.backend.write_esc)

    def center(self) -> None:
        """
        Put both channels in neutral using configured zero_us values.
        """
        s0 = int(self.servo_cfg.zero_us)
        e0 = int(self.esc_cfg.zero_us)
        self.backend.write_servo(s0)
        self.backend.write_esc(e0)
        self._last_servo_us = s0
        self._last_esc_us = e0

    def send(self, delta_rad: float, ax_cmd: float) -> Tuple[int, int]:
        """
        Send steering angle (rad) and accel command ([-1, 1]) to hardware.
        Returns the pulse widths written (servo_us, esc_us).
        """
        # Map to pulse widths
        servo_us = self.servo_map.delta_to_us(delta_rad)
        esc_us = self.esc_map.ax_to_us(ax_cmd)
        #print("Servo: ", servo_us, " ESC: ", esc_us)

        # Write (only if changed is fine; underlying backend can handle duplicates)
        if servo_us != self._last_servo_us:
            self.backend.write_servo(servo_us)
            self._last_servo_us = servo_us

        if esc_us != self._last_esc_us:
            self.backend.write_esc(esc_us)
            self._last_esc_us = esc_us

        return servo_us, esc_us

    def stop(self) -> None:
        """
        Safe shutdown: neutral outputs, then stop backend.
        """
        try:
            # Neutral before releasing hardware
            self.center()
        finally:
            self.backend.stop()
