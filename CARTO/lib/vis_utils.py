from numpy.core.multiarray import asarray
import plotly.graph_objects as go
import numpy as np
import open3d as o3d
from typing import List, Tuple
import matplotlib.pyplot as plt
import pathlib
import colorsys
from PIL import Image
import cv2


def save_image(data, file_path: pathlib.Path, FIG_DPI: int = 400, is_bgr: bool = True):
    if is_bgr:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(
        dpi=FIG_DPI, figsize=(data.shape[1] / FIG_DPI, data.shape[0] / FIG_DPI)
    )
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data)
    # plt.tight_layout()
    fig.savefig(str(file_path))
    plt.close(fig)


def draw_pcs(pcs: np.ndarray, colors_list: List[Tuple] = [], show: bool = False):
    geometries = []
    for pc in pcs:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        geometries.append(pcd)

    draw_geometries(geometries, colors_list=colors_list, show=show)


def draw_geometries(
    geometries: o3d.geometry.Geometry, colors_list=[], show: bool = False
):
    graph_objects = []

    for i, geometry in enumerate(geometries):
        geometry_type = geometry.get_geometry_type()

        if geometry_type == o3d.geometry.Geometry.Type.PointCloud:
            points = np.asarray(geometry.points)
            colors = None
            if geometry.has_colors():
                colors = np.asarray(geometry.colors)
            elif geometry.has_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.normals) * 0.5
            else:
                color_tuple = (
                    (1.0, 0.0, 0.0) if len(colors_list) == 0 else colors_list[i]
                )
                geometry.paint_uniform_color(color_tuple)
                colors = np.asarray(geometry.colors)

            scatter_3d = go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(size=1, color=colors),
            )
            graph_objects.append(scatter_3d)

        if geometry_type == o3d.geometry.Geometry.Type.TriangleMesh:
            triangles = np.asarray(geometry.triangles)
            vertices = np.asarray(geometry.vertices)
            colors = None
            if geometry.has_triangle_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.triangle_normals) * 0.5
                colors = tuple(map(tuple, colors))
            else:
                colors = (1.0, 0.0, 0.0)

            mesh_3d = go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                facecolor=colors,
                opacity=0.50,
            )
            graph_objects.append(mesh_3d)

    fig = go.Figure(
        data=graph_objects,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            )
        ),
    )
    if show:
        fig.show()
    return fig


def draw_line_w_dot(
    img, start_point, end_point, color, line_thickness: int = 1, n_lines: int = 8
):
    start_point = tuple(start_point)
    end_point = tuple(end_point)
    img_original = img.copy()
    # Linear spread
    alpha = 1.0 / (n_lines + 1)
    img = cv2.line(
        img_original.copy(), start_point, end_point, color, thickness=line_thickness
    )
    for n in range(1, n_lines + 1):
        img_alpha = cv2.line(
            img_original.copy(),
            start_point,
            end_point,
            color,
            thickness=line_thickness + n,
        )
        img = cv2.addWeighted(img, 1.0 - alpha, img_alpha, alpha, 0.0)

    img = cv2.circle(
        img,
        start_point,
        radius=int(line_thickness + n_lines / 2),
        color=color,
        thickness=-1,
    )
    img = cv2.circle(
        img,
        end_point,
        radius=int(line_thickness + n_lines / 2),
        color=color,
        thickness=-1,
    )
    return img


def draw_bboxes_glow(img, img_pts, axes, color):
    img = cv2.arrowedLine(img, tuple(axes[0]), tuple(axes[1]), (0, 0, 255), 4)
    img = cv2.arrowedLine(img, tuple(axes[0]), tuple(axes[3]), (255, 0, 0), 4)
    img = cv2.arrowedLine(
        img, tuple(axes[0]), tuple(axes[2]), (0, 255, 0), 4
    )  ## y last

    img_pts = np.int32(img_pts).reshape(-1, 2)

    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = draw_line_w_dot(img, img_pts[i], img_pts[j], color=color)
    for i, j in zip(range(4), range(4, 8)):
        img = draw_line_w_dot(img, img_pts[i], img_pts[j], color=color)
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = draw_line_w_dot(img, img_pts[i], img_pts[j], color=color)
    return img


def overlay_projected_points(color_img, pcd_array, show=False):
    # TODO Maybe this should be implemented with opencv?

    height, width, _ = color_img.shape

    # get the size in inches
    dpi = 72.0  # TODO Magic Number?
    xinch = width / dpi
    yinch = height / dpi
    fig = plt.figure(figsize=(xinch, yinch), frameon=False)
    ax = fig.add_subplot()

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.xlim((0, width))
    plt.ylim((0, height))

    ax.imshow(color_img, interpolation="none")
    # color = ['g', 'y', 'b', 'r', 'm', 'c', '#3a7c00', '#3a7cd9', '#8b7cd9', '#211249']
    for i, points_2d_mesh in enumerate(pcd_array):
        ax.scatter(points_2d_mesh[:, 0], points_2d_mesh[:, 1], s=2)  # ,color=color[i]
    ax.invert_yaxis()
    plt.axis("off")
    # fig.axes[0].margins(0)
    fig.canvas.draw()
    # fig.show()
    # color_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # color_img = color_img.reshape((height, width, 3))
    buf = fig.canvas.buffer_rgba()
    color_img = np.asarray(buf)
    return color_img


def project(K, p_3d):
    """
    Projects 3D points (`p_3d`) given a camera intrinsics `K`
    """
    projections_2d = np.zeros((2, p_3d.shape[1]), dtype="float32")
    p_2d = np.dot(K, p_3d)
    projections_2d[0, :] = p_2d[0, :] / p_2d[2, :]
    projections_2d[1, :] = p_2d[1, :] / p_2d[2, :]
    return projections_2d


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = (
            np.array(lines)
            if lines is not None
            else self.lines_from_ordered_points(self.points)
        )
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length
            )
            cylinder_segment = cylinder_segment.translate(translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a),
                    center=cylinder_segment.get_center(),
                )
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


def line_set_mesh(points_array):
    open_3d_lines = [
        [0, 1],
        [7, 3],
        [1, 3],
        [2, 0],
        [3, 2],
        [0, 4],
        [1, 5],
        [2, 6],
        # [4, 7],
        [7, 6],
        [6, 4],
        [4, 5],
        [5, 7],
    ]
    colors = random_colors(len(open_3d_lines))
    open_3d_lines = np.array(open_3d_lines)
    line_set = LineMesh(points_array, open_3d_lines, colors=colors, radius=0.001)
    line_set = line_set.cylinder_segments
    return line_set


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors


def line_set(points_array):
    open_3d_lines = [
        [0, 1],
        [7, 3],
        [1, 3],
        [2, 0],
        [3, 2],
        [0, 4],
        [1, 5],
        [2, 6],
        # [4, 7],
        [7, 6],
        [6, 4],
        [4, 5],
        [5, 7],
    ]
    # colors = [[1, 0, 0] for i in range(len(lines))]
    colors = random_colors(len(open_3d_lines))
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_array),
        lines=o3d.utility.Vector2iVector(open_3d_lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set
