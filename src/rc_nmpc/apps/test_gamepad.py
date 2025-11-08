import time
from ..hw.gamepad import XboxController
def main():
    gp = XboxController()
    if not gp.state.connected:
        print("No gamepad found. Pair via Bluetooth first.")
    try:
        while True:
            st = gp.poll()
            print(f"steer={st.steer:+.2f} accel={st.accel:+.2f} brake={st.brake:+.2f} A={st.a_pressed} B={st.b_pressed} X={st.x_pressed} conn={st.connected}", end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nDone.")
if __name__ == "__main__":
    main()
