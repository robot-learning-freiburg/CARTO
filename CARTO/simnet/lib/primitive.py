from CARTO.simnet.lib import sg
import numpy as np
import trimesh

DEFAULT_COLOR = np.array([60, 60, 60, 255], dtype=np.uint8)
CORNER_1 = np.array([255, 255, 0, 255], dtype=np.uint8)
CORNER_2 = np.array([0, 255, 255, 255], dtype=np.uint8)
CORNER_3 = np.array([255, 0, 255, 255], dtype=np.uint8)
CORNER_4 = np.array([0, 255, 0, 255], dtype=np.uint8)
CORNER_5 = np.array([255, 0, 0, 255], dtype=np.uint8)
CORNER_6 = np.array([0, 0, 255, 255], dtype=np.uint8)
CORNER_7 = np.array([255, 255, 255, 255], dtype=np.uint8)
CORNER_8 = np.array([255, 255, 0, 255], dtype=np.uint8)
RED = np.array([255, 0, 0, 255], dtype=np.uint8)
GREEN = np.array([0, 255, 0, 255], dtype=np.uint8)
BLUE = np.array([0, 0, 255, 255], dtype=np.uint8)


def make_coordinate_frame(scale=1.0, name="coord_frame_vis"):
    node = sg.Node()
    small = 0.2 * scale
    large = 1.0 * scale
    node.add_child(make_cube(large, small, small, color=RED, name=f"{name}_x"))
    node.add_child(make_cube(small, large, small, color=GREEN, name=f"{name}_y"))
    node.add_child(make_cube(small, small, large, color=BLUE, name=f"{name}_z"))
    return node


def make_cube(
    x_width=1.0, y_depth=1.0, z_height=1.0, name="cube", color=None, disable_color=False
):
    if disable_color:
        vertex_colors = None
    else:
        vertex_colors = [
            CORNER_1 if color is None else color,
            CORNER_2 if color is None else color,
            CORNER_3 if color is None else color,
            CORNER_4 if color is None else color,
            CORNER_5 if color is None else color,
            CORNER_6 if color is None else color,
            CORNER_7 if color is None else color,
            CORNER_8 if color is None else color,
        ]
    mesh = trimesh.Trimesh(
        vertices=[
            [0, 0, 0],  # 0
            [x_width, 0, 0],  # 1
            [x_width, y_depth, 0],  # 2
            [0, y_depth, 0],  # 3
            [0, 0, z_height],  # 4
            [x_width, 0, z_height],  # 5
            [x_width, y_depth, z_height],  # 6
            [0, y_depth, z_height],  # 7
        ],
        faces=[
            [0, 3, 2, 1],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [0, 4, 7, 3],
        ],
        vertex_colors=vertex_colors,
    )
    node = sg.Node(name=name)
    node.meshes = [mesh]
    node.meta.is_object = True
    return node
