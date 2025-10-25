from rc_nmpc.utils.math import wrap_angle
def test_wrap_angle():
    import numpy as np
    x = wrap_angle(3.5)
    assert -np.pi <= x <= np.pi
