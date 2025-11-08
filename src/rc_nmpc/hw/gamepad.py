from dataclasses import dataclass
from typing import Optional
import threading
import time

try:
    from evdev import InputDevice, ecodes, list_devices
except Exception:
    InputDevice = None
    ecodes = None
    list_devices = lambda: []  # noqa: E731


@dataclass
class GamepadState:
    steer: float = 0.0           # left stick X in [-1, 1]
    accel: float = 0.0           # right trigger in [0, 1]
    brake: float = 0.0           # left trigger in [0, 1]
    a_pressed: bool = False      # A button (momentary)
    x_pressed: bool = False      # X button (momentary)
    b_pressed: bool = False      # B button (momentary)
    connected: bool = False


class XboxController:
    """
    Xbox controller reader (evdev).
    - poll() reads events and updates self.state (non-threaded usage)
    - start_background(rate_hz) runs a background poller
    - snapshot() returns a thread-safe copy of the latest state
    - stop_background() stops the background poller
    """
    def __init__(self, name_hint: str = "Xbox"):
        self.dev: Optional[InputDevice] = None
        self.state = GamepadState()
        self._find(name_hint)

        # Background polling machinery
        self._bg_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()         # protects _state_copy
        self._state_copy = GamepadState()     # thread-safe snapshot container

    def _find(self, hint: str):
        for p in list_devices():
            print(p)
            try:
                d = InputDevice(p)
                if hint.lower() in (d.name or "").lower():
                    self.dev = d
                    break
            except Exception:
                pass
        self.state.connected = self.dev is not None

    def _norm(self, v, lo, hi, dz=0.05):
        c = (v - lo) / max(1, (hi - lo))
        x = 2 * c - 1
        return 0.0 if abs(x) < dz else x

    def _norm_t(self, v, lo, hi):
        c = (v - lo) / max(1, (hi - lo))
        x = max(0.0, min(1.0, c))
        return x

    def poll(self) -> GamepadState:
        """Read new controller inputs and update the public state (non-blocking)."""
        if not self.dev:
            self.state.connected = False
            return self.state

        try:
            for e in self.dev.read():
                #print(e)
                if e.type == ecodes.EV_ABS:
                    ai = self.dev.absinfo(e.code)
                    lo, hi, flat = ai.min, ai.max, ai.flat
    
                    val = e.value
                    if e.value <= flat:
                        val = lo # 0 value if within deadzone
                    
                    if e.code == ecodes.ABS_X:
                        self.state.steer = self._norm(e.value, lo, hi)
                        #print(e.value)
                    elif e.code == ecodes.ABS_GAS:
                        self.state.accel = self._norm_t(val, lo, hi)
                    elif e.code == ecodes.ABS_BRAKE:
                        self.state.brake = self._norm_t(val, lo, hi)

                elif e.type == ecodes.EV_KEY:
                    if e.code == ecodes.BTN_SOUTH:   # A button
                        self.state.a_pressed = (e.value == 1)
                    elif e.code == ecodes.BTN_EAST:  # B button
                        self.state.b_pressed = (e.value == 1)
                    elif e.code == ecodes.BTN_C: # X button
                        self.state.x_pressed = (e.value == 1)

            self.state.connected = True
        except BlockingIOError:
            # no events right now
            pass
        except Exception:
            # device likely disconnected
            self.state.connected = False

        return self.state

    # -------- Background polling --------
    def start_background(self, rate_hz: float = 200.0):
        """Start a background thread that polls evdev and maintains a thread-safe snapshot."""
        if self._bg_thread and self._bg_thread.is_alive():
            return
        self._stop_evt.clear()
        period = 1.0 / max(1.0, rate_hz)

        def _loop():
            while not self._stop_evt.is_set():
                st = self.poll()
                # Copy to thread-safe snapshot
                with self._lock:
                    self._state_copy = GamepadState(
                        steer=st.steer,
                        accel=st.accel,
                        brake=st.brake,
                        a_pressed=st.a_pressed,
                        x_pressed=st.x_pressed,
                        b_pressed=st.b_pressed,
                        connected=st.connected,
                    )
                time.sleep(period)

        self._bg_thread = threading.Thread(target=_loop, daemon=True)
        self._bg_thread.start()

    def snapshot(self) -> GamepadState:
        """Return the latest thread-safe state without touching evdev directly."""
        with self._lock:
            st = self._state_copy
            # return a new instance to avoid exposing internal reference
            return GamepadState(
                steer=st.steer,
                accel=st.accel,
                brake=st.brake,
                a_pressed=st.a_pressed,
                x_pressed=st.x_pressed,
                b_pressed=st.b_pressed,
                connected=st.connected,
            )

    def stop_background(self):
        """Stop the background poller (safe to call multiple times)."""
        try:
            self._stop_evt.set()
            if self._bg_thread:
                self._bg_thread.join(timeout=1.0)
        except Exception:
            pass
        finally:
            self._bg_thread = None
