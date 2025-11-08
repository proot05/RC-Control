"""Main runtime with simple tracker (B), manual (default), track(NMPC placeholder).

Improvements:
- Uses background threads for BOTH OAK-D (vision) and gamepad.
- Reads non-blocking snapshots to keep the control loop deterministic.
- Failsafe: if controller disconnects -> IDLE + neutral.
- Optional: OAK-D display window (runs in its own thread via OakdTracker.start_display()).
"""
import argparse, time, math
from pathlib import Path

from rc_nmpc.utils.config import load_yaml
from rc_nmpc.utils.time_sync import Rate
from rc_nmpc.hw.actuators import Actuators, PigpioBackend
from rc_nmpc.hw.mapping import ServoConfig, EscConfig
from rc_nmpc.hw.gamepad import XboxController
from rc_nmpc.perception.oakd import OakdTracker


def build_actuators(cfg_path: str) -> Actuators:
    cfg = load_yaml(cfg_path)
    servo = ServoConfig(**cfg.get("servo", {}))
    esc = EscConfig(**cfg.get("esc", {}))
    backend = PigpioBackend(
        servo_gpio=cfg.get("servo_gpio", 18),
        esc_gpio=cfg.get("esc_gpio", 19),
    )
    return Actuators(backend, servo_cfg=servo, esc_cfg=esc)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    ap = argparse.ArgumentParser()
    # Go up to project root (run this as a module from project root)
    root = Path(__file__).resolve().parents[3]
    print(Path(__file__).resolve().parents[3])
    ap.add_argument("--actuation",  default=str(root / "config" / "actuation.yaml"))
    ap.add_argument("--perception", default=str(root / "config" / "perception.yaml"))
    ap.add_argument("--track_cfg",  default=str(root / "config" / "track_simple.yaml"))
    ap.add_argument("--mode", choices=["idle","bench","manual","track","simple"], default="manual")
    ap.add_argument("--arm", action="store_true")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--no-gamepad", action="store_true")

    # NEW: optional OAK-D display (off by default)
    ap.add_argument("--display", action="store_true",
                    help="Show a live OAK-D window with detected bbox (runs in its own thread).")
    ap.add_argument("--display-rate", type=float, default=60.0,
                    help="Refresh rate for the OAK-D display thread (Hz).")
    ap.add_argument("--display-window", type=str, default="OAK-D",
                    help="Window name for the OAK-D display.")

    args = ap.parse_args()

    # ---- Hardware
    act = build_actuators(args.actuation)
    act.center(); time.sleep(0.5)
    if args.arm:
        try:
            act.arm_esc()
        except Exception as e:
            print("Arming note:", e)

    # ---- Simple tracker parameters
    ts_cfg = load_yaml(args.track_cfg)
    kp = float(ts_cfg.get("kp_steer", 1.2))
    bearing_limit = float(ts_cfg.get("bearing_limit_rad", 0.6))
    constant_ax = float(ts_cfg.get("constant_accel", 0.4))
    min_range = float(ts_cfg.get("min_range_m", 0.3))
    stop_range = float(ts_cfg.get("stop_range_m", 0.6))
    lost_timeout_ms = int(ts_cfg.get("lost_timeout_ms", 300))

    # ---- Perception (background reader)
    tracker = OakdTracker(config_path=args.perception)
    tracker.start()
    tracker.start_background(rate_hz=60.0)  # keep latest VisionObject fresh

    # Optional display thread (handled inside OakdTracker)
    if args.display:
        try:
            tracker.start_display(window_name=args.display_window, rate_hz=args.display_rate)
            print(f"[display] Started '{args.display_window}' at {args.display_rate:.1f} Hz.")
        except Exception as e:
            print(f"[display] Could not start display: {e}")

    # ---- Gamepad (background poller)
    gp = None if args.no_gamepad else XboxController()
    if gp:
        gp.start_background(rate_hz=200.0)

    rate = Rate(args.rate)
    mode = args.mode
    k = 0.0
    last_seen_ns = 0
    prev_connected = gp.snapshot().connected if gp else False

    print(f"[main] start mode: {mode}")
    try:
        while True:
            steer_cmd = 0.0
            ax_cmd = 0.0

            # --- Controller snapshot (non-blocking)
            st = gp.snapshot() if gp else None
            if st:
                # Failsafe: detect disconnect edge
                if prev_connected and not st.connected:
                    mode = "idle"
                    print("[WARN] gamepad disconnected -> IDLE")
                prev_connected = st.connected

                if st.connected:
                    if st.a_pressed:
                        mode = "manual" if mode != "manual" else mode
                        print(f"[main] A -> {mode}")
                    elif st.b_pressed:
                        mode = "simple" if mode != "simple" else mode
                        print(f"[main] B -> {mode}")
                    elif st.x_pressed:
                        mode = "idle" if mode != "idle" else mode
                        print(f"[main] X -> {mode}")
                    elif st.y_pressed:
                        mode = "track" if mode != "track" else mode
                        print(f"[main] Y -> {mode}")

                # If in MANUAL but no controller, force neutral continuously
                if (not st.connected) and (mode == "manual"):
                    act.send(0.0, 0.0)
                    rate.sleep()
                    continue

            # --- Mode logic
            if mode == "idle":
                steer_cmd, ax_cmd = 0.0, 0.0

            elif mode == "bench":
                steer_cmd = 0.3 * math.sin(k)
                ax_cmd = 0.4 * math.sin(0.5 * k)
                k += 0.05

            elif mode == "manual":
                if st and st.connected:
                    steer_cmd = st.steer * 0.6
                    ax_cmd = st.accel - st.brake

            elif mode == "track":
                # NMPC placeholder '96 to be implemented
                steer_cmd, ax_cmd = 0.0, 0.0

            elif mode == "simple":
                # Non-blocking, stale-safe vision read
                obj = tracker.get_latest(max_age_ms=200)
                t_now_ns = time.monotonic_ns()
                if obj is not None and obj.conf > 0.0:
                    last_seen_ns = obj.t_cam
                    # Bearing in horizontal plane, camera/body frame
                    bearing = math.atan2(obj.x_b, obj.z_b if obj.z_b != 0.0 else 1e-6)
                    steer_cmd = clamp(kp * bearing, -bearing_limit, +bearing_limit)

                    # Simple forward accel unless too close
                    dist = math.hypot(obj.x_b, obj.z_b)
                    if dist < stop_range or obj.z_b < min_range:
                        ax_cmd = 0.0
                    else:
                        ax_cmd = constant_ax
                else:
                    dt_ms = (t_now_ns - last_seen_ns) / 1e6 if last_seen_ns else 1e9
                    if dt_ms > lost_timeout_ms:
                        steer_cmd, ax_cmd = 0.0, 0.0
                    else:
                        steer_cmd, ax_cmd = 0.0, 0.0

            # ---- Actuate & pace
            #print("steer_cmd: ", steer_cmd, " drive: ", ax_cmd)
            act.send(steer_cmd, ax_cmd)
            rate.sleep()

    except KeyboardInterrupt:
        print("n[main] Stopping.")
    finally:
        try:
            tracker.close()  # also stops display thread if running
        except Exception:
            pass
        if gp:
            gp.stop_background()
        act.stop()


if __name__ == "__main__":
    main()
