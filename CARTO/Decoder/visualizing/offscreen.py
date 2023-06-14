from typing import List, Optional
import pyrender
import torch
import numpy as np


def look_at(
    center: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 1.0, 0.0])
):
    """
    params:
      center: Camera position
      target: Target to look at
      up: up axis of camera
    """

    f = center - target
    f /= np.linalg.norm(f)
    up /= np.linalg.norm(up)
    r = np.cross(up, f)
    u = np.cross(f, r)

    m = np.zeros((4, 4))
    m[0:3, 0] = r
    m[0:3, 1] = u
    m[0:3, 2] = f
    m[0:3, 3] = center
    m[3, 3] = 1.0
    return m


def get_default_scene():
    scene = pyrender.Scene()
    cam = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
    cam_pose = look_at(
        np.array([-1.0, -1.0, 1.0]),
        np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 0.0, 1.0]),
    )
    scene.add(cam, pose=cam_pose)

    light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=3.0,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 6.0,
    )
    scene.add(light, pose=cam_pose)
    return scene


def get_point_cloud(
    points: np.ndarray,
    sdf: np.ndarray,
    color: np.ndarray = np.array([0.0, 0.0, 0.0]),
    threshold: float = 0e-3,
) -> pyrender.Mesh:
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(sdf, torch.Tensor):
        sdf = sdf.cpu().numpy()
    if sdf.ndim == 2:
        sdf = sdf[:, 0]

    if not np.count_nonzero(sdf <= threshold):
        threshold = sdf.min() + 1e-5

    points = points[sdf <= threshold]
    # colors = np.ones(points.shape) * color
    colors = np.abs(points) / 2.0
    cloud = pyrender.Mesh.from_points(points, colors=colors)
    return cloud


def render_offscreen(
    scene: Optional[pyrender.Scene] = None, meshes: Optional[List[pyrender.Mesh]] = []
):
    if not scene:
        scene = get_default_scene()
    for mesh in meshes:
        scene.add(mesh)
    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
    color, depth = r.render(scene)
    r.delete()
    return color
