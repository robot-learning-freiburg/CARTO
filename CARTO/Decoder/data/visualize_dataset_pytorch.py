import argparse
import pathlib
import numpy as np
import open3d as o3d
import random
import tqdm

from CARTO.Decoder.data import dataset
from CARTO.Decoder import utils, config


def main(args):
    split_dicts = dataset.get_dataset_split_dict(
        pathlib.Path(args.data_dir), args.split_name, file_name=args.split_file_name
    )
    gen_cfg: config.GenerationConfig = utils.load_cfg(
        pathlib.Path(args.data_dir), cfg_class=config.GenerationConfig
    )
    rescaler = dataset.Rescaler3D(scale=gen_cfg.max_extent)
    print(gen_cfg.max_extent)
    train_dataset = dataset.SDFDataset(
        split_dicts["train"], rescaler=rescaler, cache_in_ram=False, subsample=100000000
    )
    val_dataset = dataset.SDFDataset(
        split_dicts["val"], rescaler=rescaler, cache_in_ram=False
    )

    print(f"{len(train_dataset) = }")

    pcds = []
    k = 100
    # k = len(train_dataset)
    indices = random.sample(range(len(train_dataset)), k)

    for i in tqdm.tqdm(indices):
        data_point: dataset.DataPoint = train_dataset[i]

        sdf = data_point.sdf_values
        points = data_point.points[sdf <= 0.0]
        color = utils.get_random_color()

        if np.abs(points).max() > 1:
            print(np.abs(points).max())

        # print(points.shape)
        # print(color)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)
        pcds.append(pcd)

    pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame())
    print(len(pcds))

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

    o3d.visualization.draw_geometries(pcds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("split_name")
    parser.add_argument("--unit-cube", action="store_true", default=True)
    args = parser.parse_args()
    main(args)
