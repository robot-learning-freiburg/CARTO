import dataclasses

import numpy as np

X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])


@dataclasses.dataclass
class Pose:
    camera_T_object: np.ndarray
    scale_matrix: np.ndarray = np.eye(4)


class Transform:
    def __init__(self, matrix=None):
        if matrix is None:
            self.matrix = np.eye(4)
        else:
            self.matrix = matrix
        self.is_concrete = True

    def apply_transform(self, transform):
        assert self.is_concrete
        assert isinstance(transform, Transform)
        self.matrix = self.matrix @ transform.matrix

    def inverse(self):
        assert self.is_concrete
        return Transform(matrix=np.linalg.inv(self.matrix))

    def __repr__(self):
        assert self.matrix.shape == (4, 4)
        if self.is_SE3():
            return f"Transform(trans={self.translation.tolist()}, rot={self.rotation.tolist()})"
        else:
            return f"Transform(IS_NOT_SE3,matrix={self.matrix})"

    def is_SE3(self):
        return matrixIsSE3(self.matrix)

    @property
    def translation(self):
        return self.matrix[:3, 3]

    @translation.setter
    def translation(self, value):
        assert value.shape == (3,)
        self.matrix[:3, 3] = value

    @property
    def rotation(self):
        return self.matrix[:3, :3]

    @rotation.setter
    def rotation(self, value):
        assert value.shape == (3, 3)
        self.matrix[:3, :3] = value

    @classmethod
    def from_aa(cls, axis=X_AXIS, angle_deg=0.0, translation=None):
        assert axis.shape == (3,)
        matrix = np.eye(4)
        if angle_deg != 0.0:
            matrix[:3, :3] = axis_angle_to_rotation_matrix(axis, np.deg2rad(angle_deg))
        if translation is not None:
            translation = np.array(translation)
            assert translation.shape == (3,)
            matrix[:3, 3] = translation
        return cls(matrix=matrix)


def matrixIsSE3(matrix):
    if not np.allclose(matrix[3, :], np.array([0.0, 0.0, 0.0, 1.0])):
        return False
    rot = matrix[:3, :3]
    if not np.allclose(rot @ rot.T, np.eye(3)):
        return False
    if not np.isclose(np.linalg.det(rot), 1.0):
        return False
    return True


def find_closest_SE3(matrix):
    matrix = np.copy(matrix)
    assert np.allclose(matrix[3, :], np.array([0.0, 0.0, 0.0, 1.0]))
    rotation = matrix[:3, :3]
    u, s, vh = np.linalg.svd(rotation)
    matrix[:3, :3] = u @ vh
    assert matrixIsSE3(matrix)
    return matrix


def apply_scale_transform(obj_node, scale_value):
    scale_matrix = np.eye(4)
    scale_matrix[0:3, 0:3] = scale_matrix[0:3, 0:3] * scale_value
    scale_transform = Transform(matrix=scale_matrix)
    obj_node.apply_transform(scale_transform)
    obj_node.apply_transforms_to_meshes()


def center_mesh(obj_node):
    current_scale = obj_node.bbox_lengths()
    bounding_box = obj_node.bbox()
    bounds_centroid = np.array(
        [
            current_scale[0] / 2.0 + bounding_box[0][0],
            current_scale[1] / 2.0 + bounding_box[0][1],
            current_scale[2] / 2.0 + bounding_box[0][2],
        ]
    )
    mesh_to_centroid = np.eye(4)
    mesh_to_centroid[0:3, 3] = -1.0 * bounds_centroid
    centering_transform = Transform(matrix=mesh_to_centroid)
    obj_node.apply_transform(centering_transform)
    obj_node.apply_transforms_to_meshes()


def apply_scale_matrix(obj_node, scale_matrix):
    obj_bounds = obj_node.bbox_lengths()
    # Translate to unit scale.
    unit_scaling_matrix = np.diag(
        [1.0 / obj_bounds[0], 1.0 / obj_bounds[1], 1.0 / obj_bounds[2], 1.0]
    )
    scale_transform = Transform(matrix=scale_matrix @ unit_scaling_matrix)
    obj_node.apply_transform(scale_transform)
    obj_node.apply_transforms_to_meshes()


def apply_absolute_scale_value(obj_node, scale_value):
    obj_bounds = obj_node.bbox_lengths()
    max_dim = np.max(obj_bounds)
    # Translate to unit scale.
    unit_scaling_matrix = np.diag([1.0 / max_dim, 1.0 / max_dim, 1.0 / max_dim, 1.0])
    scale_transform = Transform(matrix=(scale_value * np.eye(4)) @ unit_scaling_matrix)
    obj_node.apply_transform(scale_transform)
    obj_node.apply_transforms_to_meshes()


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)])


def normalize_transform(transform):
    transform[:3, :3] = transform[:3, :3] / np.cbrt(np.linalg.det(transform[:3, :3]))
    return transform


def axis_angle_to_rotation_matrix(axis, theta):
    """Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Args:
        axis: a list which specifies a unit axis
        theta: an angle in radians, for which to rotate around by
    Returns:
        A 3x3 rotation matrix
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


# calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(thetas):
    for ii in range(len(thetas)):
        thetas[ii] = np.deg2rad(thetas[ii])
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(thetas[0]), -np.sin(thetas[0])],
            [0, np.sin(thetas[0]), np.cos(thetas[0])],
        ]
    )

    R_y = np.array(
        [
            [np.cos(thetas[1]), 0, np.sin(thetas[1])],
            [0, 1, 0],
            [-np.sin(thetas[1]), 0, np.cos(thetas[1])],
        ]
    )

    R_z = np.array(
        [
            [np.cos(thetas[2]), -np.sin(thetas[2]), 0],
            [np.sin(thetas[2]), np.cos(thetas[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def quaternion_to_rotation_matrix(q):
    """Calculate rotation matrix corresponding to quaternion

    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows non-unit
    quaternions.

    References
    ----------
    Algorithm from
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    """
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < np.finfo(np.float).eps:
        return np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    return np.array(
        [
            [1.0 - (yY + zZ), xY - wZ, xZ + wY],
            [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
            [xZ - wY, yZ + wX, 1.0 - (xX + yY)],
        ]
    )


def compute_trimesh_centroid(obj):
    bounds = obj.bounds
    bounds_centroid = np.array(
        [
            (bounds[1][0] - bounds[0][0]) / 2.0 + bounds[0][0],
            (bounds[1][1] - bounds[0][1]) / 2.0 + bounds[0][1],
            (bounds[1][2] - bounds[0][2]) / 2.0 + bounds[0][2],
        ]
    )
    return bounds_centroid


class RotatedArticulation:
    def __init__(self, num_increments=2):
        super().__init__()
        self.rotation_axis = None
        self.angle_range = None

    def overwrite_sample(self, angle):
        return Transform.from_aa(axis=self.rotation_axis, angle_deg=angle)

    def __repr__(self):
        return f"RotatedArticulation(angle_range={self.angle_range})"
