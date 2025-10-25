from dataclasses import dataclass
from typing import Optional
from .mapping import ServoConfig, EscConfig, SteeringMapper, EscMapper
from .backends.pigpio_pwm import PigpioPwm, PigpioPins

@dataclass
class PigpioBackend:
    servo_gpio: int
    esc_gpio: int
    def __post_init__(self):
        self._impl = PigpioPwm(PigpioPins(self.servo_gpio, self.esc_gpio))
    def write_servo(self, us: int): self._impl.write_servo(us)
    def write_esc(self, us: int): self._impl.write_esc(us)
    def stop(self): self._impl.stop()

class Actuators:
    def __init__(self, backend: PigpioBackend,
                 servo_cfg: Optional[ServoConfig]=None,
                 esc_cfg: Optional[EscConfig]=None):
        self.backend = backend
        self.servo_map = SteeringMapper(servo_cfg or ServoConfig())
        self.esc_map = EscMapper(esc_cfg or EscConfig())
    def arm_esc(self):
        self.esc_map.arm_sequence(self.backend.write_esc)
    def center(self):
        self.backend.write_servo(1500); self.backend.write_esc(1500)
    def send(self, delta_rad: float, ax_cmd: float):
        s = self.servo_map.delta_to_us(delta_rad)
        e = self.esc_map.ax_to_us(ax_cmd)
        self.backend.write_servo(s); self.backend.write_esc(e)
        return s, e
    def stop(self):
        self.backend.stop()
