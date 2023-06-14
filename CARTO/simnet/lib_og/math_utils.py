import numpy as np


def align_rotation(R):
    """Align rotations for symmetric objects.
    Args:
        sRT: 4 x 4
    """

    theta_x = R[0, 0] + R[2, 2]
    theta_y = R[0, 2] - R[2, 0]
    r_norm = np.sqrt(theta_x**2 + theta_y**2)
    s_map = np.array(
        [
            [theta_x / r_norm, 0.0, -theta_y / r_norm],
            [0.0, 1.0, 0.0],
            [theta_y / r_norm, 0.0, theta_x / r_norm],
        ]
    )
    rotation = R @ s_map
    return rotation
