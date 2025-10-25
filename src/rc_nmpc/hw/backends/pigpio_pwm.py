from dataclasses import dataclass
import pigpio

@dataclass
class PigpioPins:
    servo_gpio: int
    esc_gpio: int

class PigpioPwm:
    def __init__(self, pins: PigpioPins):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon not connected. Run `sudo pigpiod`.")
        self.pins = pins
        self.pi.set_mode(pins.servo_gpio, pigpio.OUTPUT)
        self.pi.set_mode(pins.esc_gpio, pigpio.OUTPUT)
    def write_servo(self, us: int): self.pi.set_servo_pulsewidth(self.pins.servo_gpio, int(us))
    def write_esc(self, us: int): self.pi.set_servo_pulsewidth(self.pins.esc_gpio, int(us))
    def stop(self):
        self.pi.set_servo_pulsewidth(self.pins.servo_gpio, 0)
        self.pi.set_servo_pulsewidth(self.pins.esc_gpio, 0)
        self.pi.stop()
