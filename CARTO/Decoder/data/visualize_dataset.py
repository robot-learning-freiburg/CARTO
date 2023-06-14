## Load training SDFs
import argparse
import colorsys
import os
import numpy as np
import pathlib
import tqdm
import open3d as o3d
import random

from traitlets import default

from CARTO.simnet.lib.datapoint import decompress_datapoint
from CARTO.Decoder import utils
from CARTO.Decoder.data import dataset
from CARTO.Decoder import config


def main(args):
    file_dir = pathlib.Path(args.file_dir)
    dataset_cfg: config.GenerationConfig = utils.load_cfg(
        file_dir, cfg_class=config.GenerationConfig
    )
    all_files = list(file_dir.glob("*.zstd"))
    if args.latest or args.earliest:
        all_files.sort(key=lambda x: os.path.getmtime(x), reverse=args.earliest)
    else:
        print("Shuffling object list")
        random.shuffle(all_files)

    counts = utils.AccumulatorDict()
    for file_name in all_files:
        counts.increment(str(file_name).split("_")[-2], 1)
    print(counts)

    pcds = []
    object_ratios = []
    all_max = 0.0
    for i, file_path in tqdm.tqdm(enumerate(all_files[: args.n])):
        with open(file_path, "rb") as fh:
            buf = fh.read()
            data_point: dataset.DataPoint = decompress_datapoint(buf)

        if args.sdf:
            # print(data_point.keys())
            sdf = data_point.sdf_values
            points = data_point.points[sdf <= 0.0]
            color = utils.get_random_color()
            normals = None
        elif args.pc:
            points = data_point.full_pc
            normals = data_point.full_normals

        if len(points) == 0:
            continue

        all_max = max(all_max, np.max(points))
        # if (np.max(points) < 1.0):
        #   continue
        # print("Adding to Visualization")

        points /= dataset_cfg.max_extent

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if args.pc:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        if args.sdf:
            pcd.paint_uniform_color(color)
        pcds.append(pcd)

        if args.sdf:
            object_ratios.append(np.count_nonzero(sdf <= 0) / sdf.shape[0])

    if args.unit_cube:
        cube_points = np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float,
        )
        # cube_points /= 2
        lines = np.array(
            [
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 3],
                [4, 5],
                [4, 6],
                [5, 7],
                [6, 7],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]
        )
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(cube_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        pcds.append(line_set)

    pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame())
    o3d.visualization.draw_geometries(pcds)

    print(f"{all_max = }")

    if args.sdf:
        print(f"{object_ratios = }\n\tw/ mean {np.array(object_ratios).mean()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_dir")
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("-l", "--latest", action="store_true", default=False)
    parser.add_argument("-e", "--earliest", action="store_true", default=False)
    parser.add_argument("-sdf", action="store_true", default=False)
    parser.add_argument("-pc", action="store_true", default=False)
    parser.add_argument("--unit-cube", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
