import chunk
from importlib.resources import path
from matplotlib import pyplot as plt
import torch
import skimage
import numpy as np
import plyfile
import open3d as o3d
import pathlib
import trimesh
import pyrender
from PIL import Image
from typing import List, Tuple
import matplotlib as mpl

from CARTO.Decoder.visualizing import offscreen


def convert_sdf_samples_to_ply(
    points_tensor: torch.Tensor,
    sdf_tensor: torch.Tensor,
    threshold=0.0,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply, assumes equally spaced points
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/JitengMu/A-SDF/blob/ab85e7bb518f571e269ae53cf2ef497d4caab9a8/asdf/mesh.py#L86
    """
    sdf = sdf_tensor.cpu().numpy()[:, 0]
    points = points_tensor.cpu().numpy()

    N = sdf.shape[0]
    N_axis = int(np.cbrt(N))  # exact cubic root
    sdf = sdf.reshape([N_axis] * 3)

    # print(f"{points.max(axis=0) = }")

    voxel_size = (points.max(axis=0) - points.min(axis=0)) / (N_axis - 1)

    if not torch.count_nonzero(sdf_tensor < threshold):
        threshold = sdf_tensor.min() + 1e-5
        print(f"Threshold to low, setting to {threshold}")

    verts, faces, normals, values = skimage.measure.marching_cubes(
        sdf, level=threshold, spacing=voxel_size, allow_degenerate=False
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    # mesh_points = np.zeros_like(verts)
    mesh_points = verts + points.min(axis=0)
    # print(f"{mesh_points.min() = }")
    # print(f"{mesh_points.max() = }")
    # print(f"{points.min(axis=0) = }")
    # mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    # mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    # mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    return ply_data


def animate_pyrender_meshes(
    in_meshes: List[pyrender.Mesh], out_dir: pathlib.Path, duration: float = 5.0
):
    out_dir.mkdir(parents=True, exist_ok=True)

    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)

    images = []
    for i, py_mesh in enumerate(in_meshes):
        # offscreen.render_offscreen(meshes = [py_mesh]) # TODO Maybe use this?
        scene = offscreen.get_default_scene()
        scene.add(py_mesh)
        r.point_size = 2.0
        color, depth = r.render(scene)
        im = Image.fromarray(color)
        im.save(str(out_dir / f"temp_{i}.png"))
        images.append(im)

    r.delete()

    img_duration = duration * 1000 / len(images)

    images[0].save(
        out_dir / "animation.gif",
        format="GIF",
        append_images=images,
        save_all=True,
        duration=img_duration,
        loop=0,
    )


def animate_point_clouds(
    in_pcds: List[o3d.geometry.PointCloud], out_dir: pathlib.Path, duration: float = 5.0
):
    pyrender_meshes = []
    for pcd in in_pcds:
        pyrender_mesh = pyrender.Mesh.from_points(
            np.asarray(pcd.points),
            # colors=np.asarray(pcd.colors) * 255,
            normals=np.asarray(pcd.normals),
        )
        # sm = trimesh.creation.uv_sphere(radius=0.01)
        # sm.visual.vertex_colors = np.asarray(pcd.colors)
        # tfs = np.tile(np.eye(4), (len(pcd.points), 1, 1))
        # tfs[:, :3, 3] = pcd.points
        # pyrender_mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs, smooth=False)
        pyrender_meshes.append(pyrender_mesh)
    animate_pyrender_meshes(pyrender_meshes, out_dir, duration=duration)


def get_o3d_render(
    frame_width: int = 1200,
    frame_height: int = 1200,
    flip_viewing_direction: bool = False,
):
    render = o3d.visualization.rendering.OffscreenRenderer(frame_width, frame_height)
    vertical_field_of_view = 60.0  # between 5 and 90 degrees
    aspect_ratio = frame_width / frame_height  # azimuth over elevation
    near_plane = 0.1
    far_plane = 10
    fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
    render.scene.camera.set_projection(
        vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type
    )
    center = [0.0, 0.0, 0.0]  # look_at target
    eye = [-1.0 if not flip_viewing_direction else 1.0, -1.0, 1.0]
    up = [0.0, 0.0, 1.0]  # camera orientation
    render.scene.camera.look_at(center, eye, up)
    render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # Black?
    # render.scene.set_background([1.0, 1.0, 1.0, 1.0]) # White?
    return render


def animate_o3d_meshes(
    in_pcds: List[o3d.geometry.PointCloud],
    out_dir: pathlib.Path,
    duration: float = 5.0,
    frame_width: int = 600,
    frame_height: int = 600,
    flip_viewing_direction: bool = False,
):
    """
    duration: time in seconds
    """
    out_dir.mkdir(exist_ok=True, parents=True)

    images = []
    render = get_o3d_render(
        frame_width=frame_width,
        frame_height=frame_height,
        flip_viewing_direction=flip_viewing_direction,
    )

    for i, pcd in enumerate(in_pcds):
        image = render_o3d_mesh(pcd, render=render)
        im = Image.fromarray(image)
        im.save(str(out_dir / f"temp_{i}.png"))
        images.append(im)

    img_duration = duration * 1000 / len(images)
    images[0].save(
        out_dir / "animation.gif",
        format="GIF",
        append_images=images,
        save_all=True,
        duration=img_duration,
        loop=0,
    )


def rotate_around_o3d_meshes(
    all_pcds,
    output_dir: pathlib.Path = None,
    colors_list: List[Tuple] = [],
    fps=10,
    duration=2,
    frame_width: int = 1200,
    frame_height: int = 1200,
    up_vector=[0.0, 0.0, 1.0],
    radius=2,
):
    # counter = 0
    images = []

    render = get_o3d_render(frame_width=frame_width, frame_height=frame_height)

    single_image_dir = output_dir / "single_images"
    single_image_dir.mkdir(parents=True, exist_ok=True)

    ### Get Center for all_pcds (first time step)
    min_points, max_points = np.inf * np.ones((3,)), -np.inf * np.ones((3,))
    for pcd in all_pcds:
        np_points = np.array(pcd.points)
        min_points = np.min((min_points, np_points.min(axis=0)), axis=0)
        max_points = np.max((max_points, np_points.max(axis=0)), axis=0)
        print(f"{min_points = }")
        print(f"{max_points = }")

    camera_center = min_points + (max_points - min_points) / 2
    print(f"{camera_center = }")

    for frame_i, rotation in enumerate(
        np.linspace(0, 2 * np.pi, endpoint=False, num=fps * duration)
    ):
        # eye = camera_center + [0., 0., 360. / rotation]  # TODO Update?
        # eye = [4., 4., 4.]
        # eye = [1., 1., 1.]

        eye = (
            camera_center + np.array([np.sin(rotation), np.cos(rotation), 0.0]) * radius
        )
        # up_vector = [0., 0., 1.]
        # print(f"--- Rendering {rotation} ---")
        # print(f"{camera_center = }")
        # print(f"{eye = }")
        # print(f"{up_vector = }")
        render.scene.camera.look_at(camera_center, eye, up_vector)

        image = render_o3d_mesh(all_pcds, render=render)
        im = Image.fromarray(image)
        im.save(single_image_dir / f"{frame_i:05d}.png")
        images.append(im)

    images[0].save(
        output_dir / "animation.gif",
        format="GIF",
        append_images=images,
        save_all=True,
        duration=duration,
        loop=0,
    )


def render_o3d_mesh(
    all_pcd: Tuple[o3d.geometry.PointCloud, List[o3d.geometry.PointCloud]],
    frame_width=1200,
    frame_height=1200,
    render: o3d.visualization.rendering.OffscreenRenderer = None,
    height_coloring=True,
    flip_viewing_direction: bool = True,
):
    if not render:
        print("Warning: Render not defined, creating new")
        render = get_o3d_render(
            frame_width=frame_width,
            frame_height=frame_height,
            flip_viewing_direction=flip_viewing_direction,
        )
    else:
        render.scene.clear_geometry()

    render.scene.set_background([1.0, 1.0, 1.0, 1.0])
    # render.scene.set_background([0.0, 0.0, 0.0, 1.0])

    # material = o3d.visualization.rendering.Material()
    material = o3d.visualization.rendering.MaterialRecord()
    # material.base_color = [1.0, 0.75, 0.0, 1.0]
    # material.base_color = [((1 / 255) * 223), ((1 / 255) * 116), ((1 / 255) * 10), 1.0]
    # material.base_color = [0.0, 0.0, 0.0, 1.0]
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    # material.point_size = 2. # Default is 3
    # print(f"{material.base_color = }")
    # material.base_color = o3d.utility.Vector3dVector(np.array(in_pcds[0].normals))
    material.shader = "defaultLit"

    # constant_color = np.ones((np.array(pcd.points).shape[0], 3)) * np.array([1.0, 0.75, 0.0])
    # pc_color = constant_color

    if not isinstance(all_pcd, list):
        all_pcd = [all_pcd]

    for idx, pcd in enumerate(all_pcd):
        if height_coloring:
            points = np.array(pcd.points)
            # assign colors to the pointcloud file
            cmap_norm = mpl.colors.Normalize(
                vmin=points[:, 2].min(), vmax=points[:, 2].max()
            )
            #'hsv' is changeable to any name as stated here: https://matplotlib.org/stable/tutorials/colors/colormaps.html
            pc_color = plt.get_cmap("jet")(cmap_norm(np.asarray(points)[:, 2]))[:, 0:3]

            pcd.estimate_normals()  # TODO Move out maybe?
            pcd.colors = o3d.utility.Vector3dVector(pc_color)

        render.scene.add_geometry(f"pcd_{idx}", pcd, material)

    # Coordinate Frame
    # render.scene.add_geometry(
    #     "coordinate_frame",
    #     o3d.geometry.TriangleMesh.create_coordinate_frame(
    #         size=0.3, origin=[-0.51307064, -0.11901558, 2.38963232]
    #     ), material
    # )

    img = render.render_to_image()
    np_image = np.asarray(img)  # [::2, ::2, :]
    # images.append(Image.fromarray((np_image * 255).astype(np.uint8)))
    return np_image


def animate_plys(
    in_directory: pathlib.Path, out_dir: pathlib.Path, duration: float = 5.0
):
    """
    DEPRECATED
    Duration: time in seconds
    """
    all_files = list(in_directory.glob("*.ply"))
    pyrender_meshes = []
    for file in sorted(all_files):
        object = trimesh.load(str(file))
        pyrender_mesh = pyrender.Mesh.from_trimesh(object)
        pyrender_meshes.append(pyrender_mesh)

    animate_pyrender_meshes(pyrender_meshes, out_dir, duration=duration)
