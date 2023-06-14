import torch

torch.cuda.set_device(0)
import dataclasses
import pathlib
import random
from typing import Optional

# Ensure mesh_to_sdf is imported first
import numpy as np
import open3d as o3d
import seaborn as sns
import torch
import tyro

from CARTO.Encoder.inference import CARTO, CARTOPrediction
from CARTO.lib.real_data import RealDataset
from CARTO.simnet.lib.net.dataset import Dataset, PanopticOutputs

sns.set()

import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import tqdm


def save_image(data, file_path: pathlib.Path, FIG_DPI: int = 400):
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


@dataclasses.dataclass
class InferenceConfig:
    model_id: str = "14i8yfym"
    checkpoint_id: Optional[str] = None
    seed: int = 12345
    start_lod: int = 4
    end_lod: int = 7
    dataset_path: Optional[str] = None
    real_data: bool = (
        False  # Real data has a larger image size than synthetic/simulated data
    )
    single_sample: int = -1
    max_samples: int = 1000


def main(cfg: InferenceConfig):
    carto = CARTO(cfg.model_id, checkpoint_id=cfg.checkpoint_id)

    if cfg.real_data:
        dataset = RealDataset("datasets/real", load_pc=False)
        dataset_name = "real"
    else:
        if cfg.dataset_path is None:
            cfg.dataset_path = carto.hparams.test_path

        dataset_name = "_".join(cfg.dataset_path.split("/")[1:])
        dataset = Dataset(cfg.dataset_path, carto.hparams)
        print(f"{len(dataset)} samples @ {cfg.dataset_path}")

    iterator = (
        tqdm.tqdm(range(cfg.max_samples))
        if cfg.single_sample < 0
        else [cfg.single_sample]
    )
    for sample_id in iterator:
        vis_dir = carto.model_dir / "vis" / dataset_name / f"full_scene_{sample_id}"

        sample: PanopticOutputs
        if cfg.real_data:
            sample, _ = dataset[sample_id]
            sample.stereo_imgs[0] = sample.stereo_imgs[0][:, ::2, ::2]
            if len(sample.depth) > 0:
                sample.depth[0].depth_pred = sample.depth[0].depth_pred[::2, ::2]
        else:
            sample = dataset[sample_id]

        carto_prediction: CARTOPrediction = carto(sample)
        carto_prediction.set_vis_dir(vis_dir)

        carto_prediction.save_rgb()
        carto_prediction.save_segmentation()
        if not carto.hparams.model_rgbd:
            carto_prediction.save_depth()
        carto_prediction.save_bbox()

        carto_prediction.save_heatmap()
        carto_prediction.save_poses()

        ply_objects = carto_prediction.get_canonical_objects(ply=True)
        for idx, ply_object in enumerate(ply_objects):
            o3d.io.write_point_cloud(
                str(vis_dir / f"predicted_pc_{idx:03d}.ply"), ply_object
            )

        #### TODO Add Function in carto_prediction?
        # for shape_id in range(len(latent_embeddings_shape)):
        #   artciulated_vis_dir = vis_dir / "articulated" / str(shape_id)
        #   artciulated_vis_dir.mkdir(exist_ok=True, parents=True)
        #   shape_code = latent_embeddings_shape[shape_id]
        #   # Overwrite a single joint state
        #   joint_dict_result = joint_decoder(torch.Tensor(latent_embeddings_arti).cuda())
        #   pred_joint_state = joint_dict_result["state"][0].detach().cpu()
        #   pred_joint_type = utils.get_joint_type_batch(joint_dict_result["type"])[0]

        #   latent_embeddings_arti = joint_embedding.poly_fits[pred_joint_type].linspace(60)
        #   latent_embeddings_shapes = np.tile(shape_code, (latent_embeddings_arti.shape[0], 1))

        #   ply_objects = decoder.get_ply_meshes(
        #       latent_embeddings_shapes,
        #       latent_embeddings_arti,
        #       distance_threshold=1e-2,
        #       lod_start=4,
        #       lod_current=8,
        #       estimate_normals=False,
        #       chunk_size=5e5
        #   )

        #   for idx, ply_object in enumerate(ply_objects):
        #     o3d.io.write_point_cloud(str(artciulated_vis_dir / f"{idx:03d}.ply"), ply_object)
        #### TODO Add Function

        carto_prediction.save_2d_points()
        carto_prediction.save_pred_obb()

        pose_dicts = {
            "abs_pose_output": carto_prediction.get_poses(),
            "root_T_camera": sample.val_data[0].root_T_camera,
        }

        save_name = str(vis_dir / f"abs_pose_output.pkl")
        with open(save_name, "wb") as output:
            pickle.dump(pose_dicts, output)

        #### TODO Add function to plot in embedding
        # carto_prediction.save_in_embeddings(...)
        ####


if __name__ == "__main__":
    cfg: InferenceConfig = tyro.parse(InferenceConfig)
    torch.random.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    main(cfg)
