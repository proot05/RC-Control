from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

TimeNs = int

@dataclass
class ImuSample:
    t: TimeNs
    accel_b: Tuple[float, float, float]
    gyro_b: Tuple[float, float, float]
    mag_b: Tuple[float, float, float]

@dataclass
class VisionObject:
    t_cam: TimeNs
    x_b: float; y_b: float; z_b: float
    conf: float = 1.0

@dataclass
class EstimatorState:
    t: TimeNs
    X: float; Y: float; psi: float
    vx: float; vy: float; r: float
    b_gyro_z: float = 0.0; b_ax: float = 0.0; b_ay: float = 0.0
    cov_diag: Optional[np.ndarray] = None
    valid: bool = True

@dataclass
class ControlCmd:
    t: TimeNs
    ax_cmd: float
    delta_cmd: float
