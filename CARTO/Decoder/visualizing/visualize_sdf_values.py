## Load training SDFs
import argparse
import colorsys
import os
import numpy as np
import pathlib
import tqdm
import open3d as o3d
import random

from CARTO.simnet.lib.datapoint import decompress_datapoint
from CARTO.Decoder import utils
from CARTO.Decoder.data import dataset
from CARTO.Decoder import config
from CARTO.Decoder.visualizing import code_vis
from PIL import Image

import seaborn as sns


def main(args):
    file_dir = pathlib.Path(args.file_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
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

    render = code_vis.get_o3d_render(frame_width=600, frame_height=600)

    for i, file_path in tqdm.tqdm(enumerate(all_files[: args.n])):
        with open(file_path, "rb") as fh:
            buf = fh.read()
            data_point: dataset.DataPoint = decompress_datapoint(buf)

            # print(data_point.keys())
        sdf = data_point.sdf_values[:, None]
        points = data_point.points
        # Assign inside/outside color
        colors = np.where(
            sdf < 0.0,
            np.ones_like(points) * sns.color_palette("tab10")[0],
            np.ones_like(points) * sns.color_palette("tab10")[1],
        )

        if len(points) == 0:
            continue

        points /= dataset_cfg.max_extent

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        img_np = code_vis.render_o3d_mesh(pcd, height_coloring=False, render=render)
        img_PIL = Image.fromarray(img_np)
        img_PIL.save(str(out_dir / f"{i}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_dir")
    parser.add_argument("out_dir")
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("-l", "--latest", action="store_true", default=False)
    parser.add_argument("-e", "--earliest", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
