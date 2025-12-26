# RC Control

Architecture:
- `perception/` (OAK-D Pro pipeline stubs)
- `estimation/` (EKF with OOS update stubs)
- `models/` (dynamic bicycle + Pacejka stubs)
- `control/` (NMPC CasADi stubs)
- `hw/` (pigpio actuators, Xbox gamepad via evdev)
- `utils/` (logging, buffers, time, math, config)
- `config/` (all YAML configs)
- `apps/` (entry points)
- `scripts/`, `docs/`, `tests/`, `data/`

## Quick start
- assumes [depthai]([https://docs.luxonis.com/software-v3/depthai/]) already installed
```
source .venv/bin/activate
pip install -e .
sudo apt-get install -y python3-pip pigpio
pip install numpy pyyaml evdev
sudo pigpiod
```

## Manual drive with Xbox controller (A toggles Manual/Track)
```bash
python -m rc_nmpc.apps.run_vehicle --mode manual --arm
```

## Test controller
```bash
python -m rc_nmpc.apps.test_gamepad
```

## Test actuators
```bash
python -m rc_nmpc.apps.test_actuators --servo 18 --esc 19
```

## Notes
- OAK-D perception code in `src/rc_nmpc/perception/oakd.py`.
- EKF/NMPC stubs (`estimation/ekf.py`, `control/nmpc.py`).
