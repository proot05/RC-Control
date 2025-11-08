import time, argparse
from ..hw.actuators import Actuators, PigpioBackend
from ..hw.mapping import ServoConfig, EscConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--servo", type=int, default=18)
    ap.add_argument("--esc", type=int, default=19)
    args = ap.parse_args()
    act = Actuators(PigpioBackend(args.servo, args.esc), ServoConfig(), EscConfig())
    act.center(); time.sleep(1.0)
    try: act.arm_esc()
    except Exception as e: print("Arming note:", e)
    for d in [-0.6,-0.3,0,0.3,0.6,0]: act.send(d,0); time.sleep(1.0)
    for a in [0,0.5,1,0,-0.3,0]: act.send(0,a); time.sleep(1.0)
    act.stop()

if __name__ == "__main__":
    main()
