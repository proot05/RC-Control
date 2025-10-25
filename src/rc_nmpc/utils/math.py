import numpy as np
def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi
def body_to_world(vx, vy, psi):
    c, s = np.cos(psi), np.sin(psi)
    return vx*c - vy*s, vx*s + vy*c
