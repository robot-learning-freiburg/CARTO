import copy

import numpy as np
import pyrender
import IPython

from CARTO.simnet.lib import transform


from CARTO.simnet.lib.net.post_processing import epnp
import open3d as o3d

CLIPPING_PLANE_NEAR = 0.4

SCALE_FACTOR = 4


class KITTICamera:
    def __init__(self, height=320, width=1120):
        # This is to go from mmt to pyrender frame
        self.RT_matrix = transform.Transform.from_aa(
            axis=transform.X_AXIS, angle_deg=180.0
        ).matrix

        self.height = height  # np.random.randint(370, 377) // 2 * 2
        self.width = width  # 384 #np.random.randint(1224, 1243) // 2 * 2

        self.stereo_baseline = np.random.uniform(0.530, 0.54)

        f = np.random.uniform(707.0, 721.5)
        cu = self.width / 2.0 - np.random.uniform(7.9, 18.6)
        cv = self.height / 2.0 - np.random.uniform(1.9, 14.6)

        self._set_intrinsics(
            np.array(
                [
                    [721.5, 0.0, 609, 4.485728e01],
                    [0.0, 721.5, 172.8, 2.163791e-01],
                    [0.0, 0.0, 1.0, 2.745884e-03],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )

    def _set_intrinsics(self, intrinsics_matrix):
        assert intrinsics_matrix.shape[0] == 4
        assert intrinsics_matrix.shape[1] == 4

        self.K_matrix = intrinsics_matrix
        self.proj_matrix = self.K_matrix @ self.RT_matrix

    def get_pyrender_camera(self):
        return pyrender.camera.IntrinsicsCamera(
            self.K_matrix[0][0],
            self.K_matrix[1][1],
            self.K_matrix[0][2],
            self.K_matrix[1][2],
            znear=0.01,
            zfar=1000.0,
        )

    def project(self, points):
        """Project 4d homogenous points (4xN) to 4d homogenous pixels (4xN)"""
        assert len(points.shape) == 2
        assert points.shape[0] == 4
        return self.proj_matrix @ points

    def deproject(self, pixels, use_RT: bool = True):
        """Deproject 4d homogenous pixels (4xN) to 4d homogenous points (4xN)"""
        assert len(pixels.shape) == 2
        assert pixels.shape[0] == 4
        if use_RT:
            return np.linalg.inv(self.proj_matrix) @ pixels
        else:
            return np.linalg.inv(self.K_matrix) @ pixels

    def splat_points(self, hpoints_camera):
        """Project 4d homogenous points (4xN) to 4d homogenous points (4xN)"""
        assert len(hpoints_camera.shape) == 2
        assert hpoints_camera.shape[0] == 4
        hpixels = self.project(hpoints_camera)
        pixels = convert_homopixels_to_pixels(hpixels)
        depths_camera = convert_homopoints_to_points(hpoints_camera)[2, :]
        image = np.zeros((self.height, self.width))
        pixel_cols = np.clip(np.round(pixels[0, :]).astype(np.int32), 0, self.width - 1)
        pixel_rows = np.clip(
            np.round(pixels[1, :]).astype(np.int32), 0, self.height - 1
        )
        image[pixel_rows, pixel_cols] = depths_camera < CLIPPING_PLANE_NEAR
        return image

    def project_points_to_depth_img(self, hpoints_camera):
        """Project 4d homogenous points (4xN) to 4d homogenous points (4xN)"""
        assert len(hpoints_camera.shape) == 2
        assert hpoints_camera.shape[0] == 4
        hpixels = self.project(hpoints_camera)
        pixels = convert_homopixels_to_pixels(hpixels)
        depths_camera = convert_homopoints_to_points(hpoints_camera)[2, :]
        image = np.zeros((self.height, self.width))
        pixel_cols = np.clip(np.round(pixels[0, :]).astype(np.int32), 0, self.width - 1)
        pixel_rows = np.clip(
            np.round(pixels[1, :]).astype(np.int32), 0, self.height - 1
        )
        image[pixel_rows, pixel_cols] = depths_camera
        return image

    def deproject_depth_image(self, depth_image, use_RT: bool = True):
        assert depth_image.shape == (self.height, self.width)
        v, u = np.indices(depth_image.shape).astype(np.float32)
        z = depth_image.reshape((1, -1))
        pixels = np.stack([u.flatten(), v.flatten()], axis=0)
        hpixels = convert_pixels_to_homopixels(pixels, z)
        hpoints = self.deproject(hpixels, use_RT=use_RT)
        return hpoints


class HSRCamera(KITTICamera):
    def __init__(self, height=960, width=1280, scale_factor=2.0):
        # This is to go from mmt to pyrender frame
        self.RT_matrix = transform.Transform.from_aa(
            axis=transform.X_AXIS, angle_deg=180.0
        ).matrix

        self.height = int(height / scale_factor)
        self.width = int(width / scale_factor)
        f_x = 975.3 / scale_factor
        f_y = 975.3 / scale_factor
        self.stereo_baseline = 0.14

        self._set_intrinsics(
            np.array(
                [
                    [f_x, 0.0, self.width / 2.0, 0.0],
                    [0.0, f_y, self.height / 2.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )


class ZED2Camera1080p(KITTICamera):
    def __init__(self, height=1024, width=1920, scale_factor=2.0):
        # This is to go from mmt to pyrender frame
        self.RT_matrix = transform.Transform.from_aa(
            axis=transform.X_AXIS, angle_deg=180.0
        ).matrix

        self.height = int(height / scale_factor)
        self.width = int(width / scale_factor)
        f = 1050 / scale_factor
        c_x = self.width / 2
        c_y = self.height / 2
        self.stereo_baseline = 0.12

        self._set_intrinsics(
            np.array(
                [
                    [f, 0.0, c_x, 0.0],
                    [0.0, f, c_y, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )


class Primesense(KITTICamera):
    def __init__(self, height=480, width=640, scale_factor=1.0):
        # This is to go from mmt to pyrender frame
        self.RT_matrix = transform.Transform.from_aa(
            axis=transform.X_AXIS, angle_deg=180.0
        ).matrix

        self.height = int(height / scale_factor)
        self.width = int(width / scale_factor)
        f_x = 533.38 / scale_factor
        f_y = 534.65 / scale_factor

        # This camera isn't a stereo pair and instead is an active light depth sensor. We have a fake stereo pair to reduce integration overhead.
        self.stereo_baseline = 0.1

        self._set_intrinsics(
            np.array(
                [
                    [f_x, 0.0, 320.47, 0.0],
                    [0.0, f_y, 238.45, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )


class RealsenseD435Camera(KITTICamera):
    def __init__(self, height=704, width=1280, scale_factor=2.0):
        # This is to go from mmt to pyrender frame
        self.RT_matrix = transform.Transform.from_aa(
            axis=transform.X_AXIS, angle_deg=180.0
        ).matrix

        self.height = int(height / scale_factor)
        self.width = int(width / scale_factor)
        c_x = 644.097 / scale_factor
        c_y = 363.143 / scale_factor
        f_x = 643.62 / scale_factor
        f_y = 643.62 / scale_factor
        self.stereo_baseline = 0.05

        self._set_intrinsics(
            np.array(
                [
                    [f_x, 0.0, c_x, 0.0],
                    [0.0, f_y, c_y, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )


def disp_to_depth(disp, K_matrix, baseline):
    # hfov = np.deg2rad(100.)
    # width = 2560.
    # fx = 0.5 * width / np.tan(0.5 * hfov)
    # b = 100e-3  # 100mm == 10cm
    # print("Hard-Coded")
    # print(f"{fx = }")
    # print(f"{b = }")

    # Copied from below
    fx = K_matrix[0, 0]
    b = baseline
    # print("From Matrix")
    # print(f"{fx = }")
    # print(f"{b = }")
    with np.errstate(divide="ignore"):
        depth = b * fx / disp
    valid = (depth >= 0.01) & (depth <= 100.0) & np.isfinite(depth)
    depth[~valid] = 0.0
    return depth


def depth_to_disp(depth, K_matrix, baseline):
    fx = K_matrix[0, 0]
    b = baseline
    valid = (depth > 300e-10) & (depth < 10e5)
    disp = b * fx / depth
    disp[~valid] = 0.0
    return disp


class Camera:
    def __init__(
        self,
        hfov_deg=100.0,
        vfov_deg=80.0,
        height=2048,
        width=2560,
        stereo_baseline=0.10,
        enable_noise=False,
        override_intrinsics=None,
    ):
        """The default camera model to match the Basler's rectified implementation at TOT"""

        # This is to go from mmt to pyrender frame
        self.RT_matrix = transform.Transform.from_aa(
            axis=transform.X_AXIS, angle_deg=180.0
        ).matrix
        if override_intrinsics is not None:
            self._set_intrinsics(override_intrinsics)
            return
        height = height // SCALE_FACTOR
        width = width // SCALE_FACTOR
        assert height % 64 == 0
        assert width % 64 == 0

        self.height = height
        self.width = width

        self.stereo_baseline = stereo_baseline
        self.is_left = True

        hfov = np.deg2rad(hfov_deg)
        vfov = np.deg2rad(vfov_deg)
        focal_length_x = 0.5 * width / np.tan(0.5 * hfov)
        focal_length_y = 0.5 * height / np.tan(0.5 * vfov)

        focal_length = focal_length_x
        focal_length_ar = focal_length_y / focal_length_x

        self._set_intrinsics(
            np.array(
                [
                    [focal_length, 0.0, width / 2.0, 0.0],
                    [0.0, focal_length * focal_length_ar, height / 2.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )

    def _set_intrinsics(self, intrinsics_matrix):
        assert intrinsics_matrix.shape[0] == 4
        assert intrinsics_matrix.shape[1] == 4

        self.K_matrix = intrinsics_matrix
        self.proj_matrix = self.K_matrix @ self.RT_matrix

    def get_pyrender_camera(self):
        return pyrender.camera.IntrinsicsCamera(
            self.K_matrix[0][0],
            self.K_matrix[1][1],
            self.K_matrix[0][2],
            self.K_matrix[1][2],
            znear=0.01,
            zfar=1000.0,
        )

    def project(self, points):
        """Project 4d homogenous points (4xN) to 4d homogenous pixels (4xN)"""
        assert len(points.shape) == 2
        assert points.shape[0] == 4
        return self.proj_matrix @ points

    def deproject(self, pixels):
        """Deproject 4d homogenous pixels (4xN) to 4d homogenous points (4xN)"""
        assert len(pixels.shape) == 2
        assert pixels.shape[0] == 4
        return np.linalg.inv(self.proj_matrix) @ pixels

    def splat_points(self, hpoints_camera):
        """Project 4d homogenous points (4xN) to 4d homogenous points (4xN)"""
        assert len(hpoints_camera.shape) == 2
        assert hpoints_camera.shape[0] == 4
        hpixels = self.project(hpoints_camera)
        pixels = convert_homopixels_to_pixels(hpixels)
        depths_camera = convert_homopoints_to_points(hpoints_camera)[2, :]
        image = np.zeros((self.height, self.width))
        pixel_cols = np.clip(np.round(pixels[0, :]).astype(np.int32), 0, self.width - 1)
        pixel_rows = np.clip(
            np.round(pixels[1, :]).astype(np.int32), 0, self.height - 1
        )
        image[pixel_rows, pixel_cols] = depths_camera < CLIPPING_PLANE_NEAR
        return image

    def deproject_depth_image(self, depth_image):
        assert depth_image.shape == (self.height, self.width)
        v, u = np.indices(depth_image.shape).astype(np.float32)
        z = depth_image.reshape((1, -1))
        pixels = np.stack([u.flatten(), v.flatten()], axis=0)
        hpixels = convert_pixels_to_homopixels(pixels, z)
        hpoints = self.deproject(hpixels)
        return hpoints


def convert_homopixels_to_pixels(pixels):
    """Project 4d homogenous pixels (4xN) to 2d pixels (2xN)"""
    assert len(pixels.shape) == 2
    assert pixels.shape[0] == 4
    pixels_3d = pixels[:3, :] / pixels[3:4, :]
    pixels_2d = pixels_3d[:2, :] / pixels_3d[2:3, :]
    assert pixels_2d.shape[1] == pixels.shape[1]
    assert pixels_2d.shape[0] == 2
    return pixels_2d


def convert_pixels_to_homopixels(pixels, depths):
    """Project 2d pixels (2xN) and depths (meters, 1xN) to 4d pixels (4xN)"""
    assert len(pixels.shape) == 2
    assert pixels.shape[0] == 2
    assert len(depths.shape) == 2
    assert depths.shape[1] == pixels.shape[1]
    assert depths.shape[0] == 1
    pixels_4d = np.concatenate(
        [
            depths * pixels,
            depths,
            np.ones_like(depths),
        ],
        axis=0,
    )
    assert pixels_4d.shape[0] == 4
    assert pixels_4d.shape[1] == pixels.shape[1]
    return pixels_4d


def convert_points_to_homopoints(points):
    """Project 3d points (3xN) to 4d homogenous points (4xN)"""
    assert len(points.shape) == 2
    assert points.shape[0] == 3
    points_4d = np.concatenate(
        [
            points,
            np.ones((1, points.shape[1])),
        ],
        axis=0,
    )
    assert points_4d.shape[1] == points.shape[1]
    assert points_4d.shape[0] == 4
    return points_4d


def convert_homopoints_to_points(points_4d):
    """Project 4d homogenous points (4xN) to 3d points (3xN)"""
    assert len(points_4d.shape) == 2
    assert points_4d.shape[0] == 4
    points_3d = points_4d[:3, :] / points_4d[3:4, :]
    assert points_3d.shape[1] == points_3d.shape[1]
    assert points_3d.shape[0] == 3
    return points_3d


def get_2d_bbox_of_9D_box(camera_model, camera_T_object, scale_matrix):
    unit_box_homopoints = convert_points_to_homopoints(epnp._WORLD_T_POINTS.T)
    morphed_homopoints = camera_T_object @ (scale_matrix @ unit_box_homopoints)
    morphed_pixels = convert_homopixels_to_pixels(
        camera_model.proj_matrix @ morphed_homopoints
    ).T
    bbox = [
        np.array([np.min(morphed_pixels[:, 1]), np.min(morphed_pixels[:, 0])]),
        np.array([np.max(morphed_pixels[:, 1]), np.max(morphed_pixels[:, 0])]),
    ]
    return bbox


def get_3d_bbox(size, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar
    Returns:
        bbox_3d: [3, N]

    """
    bbox_3d = (
        np.array(
            [
                [+size[0] / 2, +size[1] / 2, +size[2] / 2],
                [+size[0] / 2, +size[1] / 2, -size[2] / 2],
                [-size[0] / 2, +size[1] / 2, +size[2] / 2],
                [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                [-size[0] / 2, -size[1] / 2, -size[2] / 2],
            ]
        )
        + shift
    )
    return bbox_3d


def get_3d_asymetric_bbox(size_min, size_max, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar
    Returns:
        bbox_3d: [3, N]

    """
    bbox_3d = (
        np.array(
            [
                [size_max[0], size_max[1], size_max[2]],
                [size_max[0], size_max[1], size_min[2]],
                [size_min[0], size_max[1], size_max[2]],
                [size_min[0], size_max[1], size_min[2]],
                [size_max[0], size_min[1], size_max[2]],
                [size_max[0], size_min[1], size_min[2]],
                [size_min[0], size_min[1], size_max[2]],
                [size_min[0], size_min[1], size_min[2]],
            ]
        )
        + shift
    )
    return bbox_3d


def transform_coordinates_3d(coordinates, RT):
    """
    Args:
        coordinates: [3, N]
        sRT: [4, 4]

    Returns:
        new_coordinates: [3, N]

    """
    unit_box_homopoints = convert_points_to_homopoints(coordinates)
    morphed_box_homopoints = RT @ unit_box_homopoints
    new_coordinates = convert_homopoints_to_points(morphed_box_homopoints)

    return new_coordinates


def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input:
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


def transform_pc(pose, pc, _CAMERA=None, use_camera_RT=True):
    # pose.camera_T_object[:3,3] = pose.camera_T_object[:3,3]*100
    # pose.scale_matrix[:3,:3] = pose.scale_matrix[:3,:3]*100
    # for pose, pc in zip(poses, pointclouds):
    pc_homopoints = convert_points_to_homopoints(np.copy(pc.T))

    scaled_homopoints = pose.scale_matrix @ pc_homopoints
    scaled_homopoints = convert_homopoints_to_points(scaled_homopoints).T

    # centroid = np.mean(scaled_homopoints, axis=0)
    # size = np.amax(np.abs(scaled_homopoints - centroid), axis=0)
    # box = get_3d_bbox(size, shift=centroid)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.copy(scaled_homopoints))
    # print("min bounds", pcd.get_min_bound())
    # print("max bounds", pcd.get_max_bound())
    min_bounds = np.array(pcd.get_min_bound())
    max_bounds = np.array(pcd.get_max_bound())
    box = get_3d_asymetric_bbox(min_bounds, max_bounds)

    # size = 2 * np.amax(np.abs(scaled_homopoints), axis=0)
    # box = get_3d_bbox(size)

    # size = 2 * np.amax(np.abs(scaled_homopoints), axis=0)
    # #unit_box_homopoints = convert_points_to_homopoints(epnp._WORLD_T_POINTS.T)
    # box = get_3d_bbox(size)
    unit_box_homopoints = convert_points_to_homopoints(box.T)
    if use_camera_RT == True:
        morphed_pc_homopoints = (
            _CAMERA.RT_matrix
            @ pose.camera_T_object
            @ (pose.scale_matrix @ pc_homopoints)
        )
        morphed_box_homopoints = (
            _CAMERA.RT_matrix
            @ pose.camera_T_object
            @ (pose.scale_matrix @ unit_box_homopoints)
        )
    else:
        morphed_pc_homopoints = pose.camera_T_object @ (
            pose.scale_matrix @ pc_homopoints
        )
        # morphed_box_homopoints = pose.camera_T_object @ (pose.scale_matrix @ unit_box_homopoints)
        morphed_box_homopoints = pose.camera_T_object @ unit_box_homopoints

    morphed_pc_homopoints = convert_homopoints_to_points(morphed_pc_homopoints).T
    morphed_box_points = convert_homopoints_to_points(morphed_box_homopoints).T
    # box_points.append(morphed_box_points)
    return morphed_pc_homopoints, morphed_box_points
