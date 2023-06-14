#!/opt/mmt/python_venv/bin/python

import argparse
import json
import logging
import pathlib
import random
import subprocess
from collections import defaultdict
from concurrent import futures
import tarfile

import numpy as np
import tqdm
import trimesh
import urdfpy
import zstandard as zstd
from CARTO.lib.partnet_mobility import get_joint_dict
from CARTO.lib.compression import write_compressed_json


PARALLEL = True


def identity_matrix():
    return np.eye(4)


around_z_neg_90 = trimesh.transformations.rotation_matrix(
    np.pi / 2, np.array([0.0, 0.0, -1.0])
)

# Dictionaries for transformations of objects that might be not in a canonical way!
canonical_transformations_cat = defaultdict(
    identity_matrix,
    {
        "Pliers": around_z_neg_90,
        # "Scissors": around_z_neg_90
    },
)
canonical_transformations_instance = defaultdict(
    identity_matrix, {"d01ff66659767d50cee19268a161fc4a": around_z_neg_90}
)


def main(top_dir=pathlib.Path("datasets/partnet-mobility-v0/raw_dataset")):
    model_dirs = (top_dir).glob("*")

    (top_dir / ".." / "tarfiles").mkdir(exist_ok=True, parents=True)

    full_index = []
    if PARALLEL:
        all_futures = []
        with futures.ProcessPoolExecutor() as executor:
            for model_dir in model_dirs:
                all_futures.append(executor.submit(process_model, model_dir))
            with tqdm.tqdm(total=len(all_futures)) as pbar:
                for future in futures.as_completed(all_futures):
                    pbar.update(1)
                    full_index.append(future.result())
    else:
        for model_dir in tqdm.tqdm(model_dirs, total=len(model_dirs)):
            full_index.append(process_model(model_dir))

    index = [meta for (meta, safe) in full_index if safe]
    print(f"Found {len(full_index)} models but only {len(index)} are safe")

    print("Writing index")
    index = sorted(index, key=lambda x: x["model_id"])
    index_path = top_dir / ".." / "index.json.zst"
    write_compressed_json(index, index_path)


def load_semantics(semantics_file):
    joint_meta_info = {}
    for line in semantics_file.readlines():
        line_entries = line.rstrip("\n").split(" ")
        joint_meta_info[f"joint_{int(line_entries[0].split('_')[1])}"] = {
            "sem_type": line_entries[1],
            "sem_name": line_entries[2],
        }
    return joint_meta_info


def process_model(model_dir: pathlib.Path):
    with open(model_dir / "meta.json") as fh:
        meta = json.load(fh)
    assert "model_id" in meta
    model_id = meta["model_id"]

    # Create tar-ball
    all_paths = model_dir.glob("**/*")
    tar_path = model_dir / ".." / ".." / "tarfiles" / (model_id + ".tar.zst")
    cctx = zstd.ZstdCompressor()
    with open(tar_path, "wb") as raw_fh:
        with cctx.stream_writer(raw_fh) as zst_fh:
            with tarfile.open(fileobj=zst_fh, mode="w") as tar:
                for path in all_paths:
                    rel_path = path.relative_to(model_dir)
                    tar.add(str(path), arcname=str(rel_path), recursive=False)

    tar_bytes = tar_path.stat().st_size
    meta["num_bytes"] = tar_bytes

    with open(model_dir / "semantics.txt") as fh:
        joint_semantics = load_semantics(fh)

    # Try loading the URDF
    # This step is important as PartNetMobility might miss some .obj!
    try:
        urdf = urdfpy.URDF.load(str(model_dir / "mobility.urdf"))
    except ValueError as e:
        logging.warning(f"urdfpy could not load model at {model_dir} with error\n{e}")
        return None, False

    # Manually parse relevant joint informations for saving in index
    joint: urdfpy.Joint
    for joint in urdf.joints:
        try:
            joint_dict = get_joint_dict(joint)
            joint_semantics[joint_dict["id"]].update(joint_dict)
        except:
            has_slider_plus = False
            for joint_semants in joint_semantics.values():
                has_slider_plus |= joint_semants["sem_type"] == "slider+"
            if not has_slider_plus:
                print(f"--- {model_id} @ {model_dir} ---")
                print(f"{joint_semantics = }")
                print(f"{joint_dict = }")
            else:
                print(f"-- Found 'slider+'-type")
    meta["joints"] = joint_semantics

    trans_cat = canonical_transformations_cat[meta["model_cat"]]
    trans_ins = canonical_transformations_instance[model_id]
    meta["canonical_transformation"] = (trans_cat @ trans_ins).tolist()

    return meta, True


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create index for PartnetMobility V0")
    args = parser.parse_args()
    main()
