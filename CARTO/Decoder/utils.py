import copy
import dataclasses
import itertools
import json
import logging
import operator
import os
import pathlib
import traceback
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import mesh_to_sdf
import numpy as np
import plyfile
import pyrender
import skimage
import torch
import tqdm
import trimesh
import urdfpy
import yaml

import wandb
from CARTO import ROOT_DIR
from CARTO.Decoder import config, embedding
from CARTO.Decoder.data import dataset
from CARTO.Decoder.models import SDF_decoder, joint_state_decoder, lr_schedules
from CARTO.lib import partnet_mobility, rename_unpickler
from CARTO.lib.partnet_mobility import PartNetMobilityV0DB

try:
    from CARTO.Decoder.data import asdf_dataset
except Exception as e:
    logging.critical(e, exc_info=True)  # log exception info at CRITICAL log level

import functools
import random

import tyro


def retry(func=None, *, times=100):
    """
    Decorator to retry any functions 'times' times.
    """
    if func is None:
        return functools.partial(retry, times=times)

    @functools.wraps(func)
    def retried_function(*args, **kwargs):
        for _ in range(times):
            try:
                return func(*args, **kwargs)
            except Exception as exception:
                print(
                    f"Exception: {type(exception).__name__}\nException message: {exception}\nFunction: {func.__name__}\n{args = }\n{kwargs =}"
                )
                pass

    return retried_function


def get_object_rescale(urdf_object: urdfpy.URDF, origin_frame: config.ObjectOrigin):
    if origin_frame == config.ObjectOrigin.BASE:
        base_link_name: str = ""
        joint: urdfpy.Joint
        for joint in urdf_object.joints:
            if joint.joint_type not in ["fixed", "planar", "floating"]:
                continue

            base_link_name = joint.child
            break
        assert (
            base_link_name != ""
        ), f"[{os.path.abspath(__file__)}] Couldn't find base_link_name"
        fk_results = urdf_object.visual_trimesh_fk(links=[base_link_name])
    elif origin_frame == config.ObjectOrigin.CLOSED_STATE:
        zero_joint_config = {
            joint: joint.limit.lower if joint.limit is not None else 0.0
            for joint in urdf_object.joints
        }
        fk_results = urdf_object.visual_trimesh_fk(cfg=zero_joint_config)
    else:
        fk_results = urdf_object.visual_trimesh_fk()

    all_meshes_trans = []
    for mesh, transform in fk_results.items():
        mesh_tmp = copy.deepcopy(mesh)
        mesh_tmp.apply_transform(transform)
        all_meshes_trans.append(mesh_tmp)
    concat_meshes_trans: trimesh.base.Trimesh = trimesh.util.concatenate(
        all_meshes_trans
    )
    max_extent = np.max(concat_meshes_trans.extents) / 2
    # This could also be the norm of all vertices --> rescale to unit-sphere
    # max_extent = np.norm(concat_meshes_trans.vertices, axis=1).max()
    return concat_meshes_trans.centroid, max_extent


def object_to_trimesh(
    urdf_object: urdfpy.URDF,
    joint_config={},
    base_transform=np.eye(4),
    origin_frame: config.ObjectOrigin = config.ObjectOrigin.CLOSED_STATE,
    additional_scale: float = 1.0,
) -> trimesh.Scene:
    """
    - origin_frame: config.ObjectOrigin
        Scales the object to be in its max extent as well as puts the base of the mesh in the center of the specified part
    """
    trans, max_extent = get_object_rescale(urdf_object, origin_frame)

    trans_offset = np.eye(4)
    trans_offset[:3, 3] = -trans
    # Applying this scaling here modifies the joint state scaling as well
    # Especially for prismatic joints
    trans_offset[0, 0] = 1 / max_extent
    trans_offset[1, 1] = 1 / max_extent
    trans_offset[2, 2] = 1 / max_extent

    # Only apply transformation if we want to override partnet mobility
    if origin_frame != config.ObjectOrigin.PARTNETMOBILITY:
        base_transform = trans_offset @ base_transform
    # base_transform = base_transform @ trans_offset

    add_rescale = np.eye(4)
    add_rescale[0, 0] = 1 / additional_scale
    add_rescale[1, 1] = 1 / additional_scale
    add_rescale[2, 2] = 1 / additional_scale
    base_transform = add_rescale @ base_transform

    fk = urdf_object.visual_trimesh_fk(cfg=joint_config)

    trimesh_scene = trimesh.Scene()
    for mesh, pose in fk.items():
        trimesh_scene.add_geometry(mesh, transform=base_transform @ pose)
    # This post applies the transform?
    # trimesh_scene.apply_transform(transform)
    # trimesh_scene.apply_transform(trans_offset)
    return trimesh_scene, trans, max_extent


@retry
def object_to_sdf(
    object_trimesh, number_samples: int = 50000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes a urdf_object and a (partial) joint configuration
    Returns:
      - points
      - sdf
    """
    points, sdf = mesh_to_sdf.sample_sdf_near_surface(
        object_trimesh,
        surface_point_method="scan",
        transform_back=True,
        sign_method="normal",  # The normals are mis-aligned for some PartNetMobility objects
        number_of_points=number_samples,  # default 500000
    )
    return points, sdf


@retry
def object_to_point_cloud(
    object_trimesh, number_samples: int = 1000000
) -> Tuple[np.ndarray, np.ndarray]:
    mesh, transform = mesh_to_sdf.utils.scale_to_unit_sphere(
        object_trimesh, get_transform=True
    )
    surface_point_cloud = mesh_to_sdf.get_surface_point_cloud(
        mesh,
        sample_point_count=number_samples,
        surface_point_method="sample",  # [scan, sample] To allow inside of the mesh?
        calculate_normals=True,
    )
    points = surface_point_cloud.points * transform["scale"] + transform["translation"]
    normals = surface_point_cloud.normals
    return points, normals


class IndexDict(dict):
    """
    This will incrementally count up for unseen keys
    """

    def __init__(self, counter_start: int = 0, *args, **kwargs):
        self.counter = counter_start
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            val = dict.__getitem__(self, key)
        else:
            val = self.counter
            dict.__setitem__(self, key, val)
            self.counter += 1
        return val

    # def __setitem__(self, key, val):
    #   print('SET', key, val)
    #   dict.__setitem__(self, key, val)

    # def __repr__(self):
    #   dictrepr = dict.__repr__(self)
    #   return '%s(%s)' % (type(self).__name__, dictrepr)

    # def update(self, *args, **kwargs):
    #   print('update', args, kwargs)
    #   for k, v in dict(*args, **kwargs).items():
    #     self[k] = v


class AccumulatorDict(dict):
    def __init__(self, *args, accumulator=operator.add, **kwargs):
        self.accumulator = accumulator
        self.update(*args, **kwargs)

    def increment(self, key, val):
        """
        This will increment the value for a given key
        """
        if dict.__contains__(self, key):
            val = self.accumulator(dict.__getitem__(self, key), val)
        dict.__setitem__(self, key, val)

    def increment_dict(self, other: Dict):
        for key, value in other.items():
            self.increment(key, value)


def save_checkpoint(
    experiment_directory: Union[pathlib.Path, Any],
    model: torch.nn.Module,
    joint_state_decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    object_embedding: torch.nn.Embedding,
    joint_configuration_embedding: torch.nn.Embedding,
    epoch: int,
    current_epoch: bool = False,
    latest: bool = True,
    best: bool = False,
):
    if not isinstance(experiment_directory, pathlib.Path):
        experiment_directory = pathlib.Path(experiment_directory)
    experiment_directory /= "checkpoints"
    experiment_directory.mkdir(parents=True, exist_ok=True)

    ckpt_names = []
    if current_epoch:
        ckpt_names.append(f"{epoch}")
    if latest:
        ckpt_names.append("latest")
    if best:
        ckpt_names.append("best")

    for ckpt_name in ckpt_names:
        torch.save(
            {
                "model_parameters": model.state_dict(),
                "joint_state_model_parameters": joint_state_decoder.state_dict(),
                "optimizer_parameters": optimizer.state_dict(),
                "object_embedding_parameters": object_embedding.state_dict(),
                "join_configuration_embedding_parameters": joint_configuration_embedding.state_dict(),
                "epoch": epoch,
            },
            experiment_directory / (ckpt_name + ".ckpt"),
        )


def load_checkpoint(
    experiment_directory: Union[pathlib.Path, Any],
    latest: bool = True,
    epoch=0,
    **kwargs,
):
    if not isinstance(experiment_directory, pathlib.Path):
        experiment_directory = pathlib.Path(experiment_directory)
    experiment_directory /= "checkpoints"
    ckpt_name = "latest" if latest else f"{epoch}"
    full_dict = torch.load(
        experiment_directory / (ckpt_name + ".ckpt"),
        pickle_module=rename_unpickler,
        **kwargs,
    )
    return (
        full_dict["model_parameters"],
        full_dict["joint_state_model_parameters"],
        full_dict["optimizer_parameters"],
        full_dict["object_embedding_parameters"],
        full_dict["join_configuration_embedding_parameters"],
    )


def save_cfg(
    experiment_directory: Union[pathlib.Path, Any], arti_config: config.ExperimentConfig
):
    if not isinstance(experiment_directory, pathlib.Path):
        experiment_directory = pathlib.Path(experiment_directory)
    experiment_directory /= "runs"
    experiment_directory /= str(arti_config.local_experiment_id)
    experiment_directory.mkdir(parents=True, exist_ok=True)
    with open(experiment_directory / "config.yaml", "w") as file:
        file.write(tyro.to_yaml(arti_config))
    return experiment_directory


def save_index(
    index: IndexDict, experiment_directory: Union[pathlib.Path, Any], name="index"
):
    if not isinstance(experiment_directory, pathlib.Path):
        experiment_directory = pathlib.Path(experiment_directory)
    experiment_directory /= "indices"
    experiment_directory.mkdir(parents=True, exist_ok=True)
    string_dict = {str(key): value for key, value in index.items()}
    with open(experiment_directory / f"{name}.json", "w") as file:
        json.dump(string_dict, file)


def save_dict(
    index: Dict, experiment_directory: Union[pathlib.Path, Any], name="joint_X"
):
    if not isinstance(experiment_directory, pathlib.Path):
        experiment_directory = pathlib.Path(experiment_directory)
    experiment_directory /= "dicts"
    experiment_directory.mkdir(parents=True, exist_ok=True)
    with open(experiment_directory / f"{name}.yaml", "w") as file:
        yaml.dump(index, file)


def load_index(experiment_directory: Union[pathlib.Path, Any], name="index"):
    if not isinstance(experiment_directory, pathlib.Path):
        experiment_directory = pathlib.Path(experiment_directory)
    experiment_directory /= "indices"
    with open(experiment_directory / f"{name}.json", "r") as file:
        return json.load(file)


def load_cfg(
    experiment_directory: Union[pathlib.Path, Any],
    name: str = "config.yaml",
    cfg_class: Type[dataclasses.dataclass] = config.ExperimentConfig,
) -> Type[dataclasses.dataclass]:
    with open(experiment_directory / name, "r") as file:
        cfg = tyro.from_yaml(cfg_class, file.read())
    return cfg


def load_dict(experiment_directory: Union[pathlib.Path, Any], name="joint_X"):
    if not isinstance(experiment_directory, pathlib.Path):
        experiment_directory = pathlib.Path(experiment_directory)
    experiment_directory /= "dicts"
    if not (experiment_directory / f"{name}.yaml").exists():
        return False
    with open(experiment_directory / f"{name}.yaml", "r") as file:
        return yaml.load(file, yaml.Loader)


def get_flat_voxels(dim_per_axis=[1, 1, 1], N_per_axis=256):
    stacked_axes = torch.meshgrid(
        [torch.linspace(-dim, dim, N_per_axis) for dim in dim_per_axis]
    )
    flat_voxels = torch.vstack(
        [torch.flatten(stacked_axis) for stacked_axis in stacked_axes]
    ).T
    return flat_voxels


def stack_ordered(tensor: torch.Tensor, n: int) -> torch.Tensor:
    """
    Takes as input a tensor of size `[b_1, b_2, ..., b_B]xD` and
    repeats it as `[b_1*n, b_2*n, ..., b_B*n]xD`
    """
    # TODO Nick: might just replace this with torch.repeat_interleave?!
    B, D = tensor.size()
    assert tensor.ndim == 2
    return tensor.repeat((1, n)).view(B * n, D)


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int):
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["scheduler"].get_learning_rate(epoch)


def svd_projection_np(data: np.ndarray, dim=2):
    U, S, Vh = np.linalg.svd(data)
    return data @ Vh[:dim, :].T


def torch_tensor_to_2_dim_wandb(
    data: torch.Tensor, projector: Callable[[np.ndarray], np.ndarray]
):
    data_np = data.cpu().detach().numpy()
    assert data_np.ndim == 2
    projected_data = projector(data_np)
    assert projected_data.shape[1] == 2 and projected_data.ndim == 2
    table = wandb.Table(data=projected_data, columns=["x", "y"])
    return wandb.plot.scatter(table, "x", "y")


def distance_to_sim(x: torch.Tensor):
    """
    Segaran, T. 2007. Programming Collective Intelligence. O’Reilly Media.
    """
    return 1 / (1 + x)


def exp_kernel(x: Union[np.ndarray, float], variance: float = 1.0):
    """
    Be careful, this does NOT square the input and assumes it already is

    Implementation: `np.exp(-x / variance**2)`
    """
    return np.exp(-x / variance**2)


def self_cosine_similarity(a):
    # TODO Nick: Check that this is correct
    assert a.ndim == 2
    norm = torch.norm(a, dim=1, keepdim=True)
    norm = norm @ norm.T
    b = a.repeat(1, 1)
    c = b @ b.T
    c /= norm
    return c


def self_manhattan_distance(a):
    assert a.ndim == 2
    sub = a[None, ...] - a[:, None, ...]
    return torch.sum(torch.abs(sub), dim=-1)


def self_euclidean_distance(a, eps=0.0):
    assert a.ndim == 2
    sub = a[None, ...] - a[:, None, ...]
    return torch.norm(sub, dim=-1)


def leaky_clamp(x, cut_off, slope):
    return torch.where(
        torch.abs(x) < cut_off,
        x,
        torch.sign(x) * cut_off + (x - torch.sign(x) * cut_off) * slope,
    )


def get_zero_joint_dict(
    joint_state_dict: Dict[str, float],
    joint_definition_dict: Dict[str, Dict[str, Any]],
    prismatic_scale: float = 1.0,
):
    """
    Returns a copy of the original joint dict but all joint states start from 0.
    Basically for each entry we are doing: joint.state - joint.limit.lower
    # Assumes joint_definition dict is already scaled!
    """
    zero_joint_config = {}
    for joint_id, state in joint_state_dict.items():
        new_state = (
            state
            / (
                prismatic_scale
                if joint_definition_dict[joint_id]["type"] == "prismatic"
                else 1.0
            )
        ) - joint_definition_dict[joint_id]["limit"][0]

        if (
            new_state < 0 - 1e-5
            or new_state
            > (
                joint_definition_dict[joint_id]["limit"][1]
                - joint_definition_dict[joint_id]["limit"][0]
            )
            + 1e-5
        ):
            print(" -- Warning -- ")
            print(" -- This might happened because of a legacy dataset -- ")
            print(f"{state} --> {new_state}")
            print(
                f'{joint_definition_dict[joint_id]["limit"] = } --> {joint_definition_dict[joint_id]["limit"][1] - joint_definition_dict[joint_id]["limit"][0]}'
            )

        zero_joint_config[joint_id] = new_state

    return zero_joint_config


def get_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return np.array([r, g, b]) / 255.0


def linspace_circle_(n=50, center: np.ndarray = np.zeros((2,)), r: float = 1.0):
    """
    Returns n evenly spaced x, y coordinates for a circle given a center and radius
    https://stackoverflow.com/a/33530407
    """
    for j in range(n):
        t = j * (2 * np.pi / n)
        xy = np.array([r * np.cos(t), r * np.sin(t)])
        yield xy + center


def linspace_circle(**kwargs):
    return np.array(list(linspace_circle_(**kwargs)))


def load_full_decoder(training_id, additional_outputs={}, to_cuda=True):
    local_dir = config.BASE_DIR / "runs" / training_id
    arti_cfg: config.ExperimentConfig = load_cfg(local_dir)
    if "cfg" in additional_outputs:
        additional_outputs["cfg"] = arti_cfg

    (
        model_params,
        joint_state_model_params,
        _,
        object_embedding_parameters,
        joint_config_parameters,
    ) = load_checkpoint(
        local_dir,
        map_location=torch.device(
            "cpu"
        ),  # , latest=False, epoch="best" # Default: loading latest checkpoint, these two arguments replace that with best
    )
    object_index = load_index(local_dir, "object")
    joint_config_index = load_index(local_dir, "joint_config")
    full_index = load_index(local_dir, "full")

    if (
        "train_dataset" in additional_outputs.keys()
        or "test_dataset" in additional_outputs.keys()
    ):
        if arti_cfg.dataset == config.DatasetChoice.ours:
            split_dicts = dataset.get_dataset_split_dict(
                ROOT_DIR / ".." / pathlib.Path(arti_cfg.training_data_dir),
                arti_cfg.split_name,
                file_name=arti_cfg.split_file_name,
            )

            gen_cfg: config.GenerationConfig = load_cfg(
                ROOT_DIR / ".." / pathlib.Path(arti_cfg.training_data_dir),
                cfg_class=config.GenerationConfig,
            )
            rescaler = dataset.Rescaler3D(scale=gen_cfg.max_extent)
            train_dataset = dataset.SimpleDataset(
                split_dicts["train"], rescaler=rescaler
            )
            # val_dataset = dataset.SimpleDataset(split_dicts["val"], rescaler=rescaler)
            test_dataset = dataset.SimpleDataset(split_dicts["test"], rescaler=rescaler)
        elif arti_cfg.dataset == config.DatasetChoice.asdf:
            assert (
                arti_cfg.scene_decoder == config.SceneDecoder.SDF
            ), "ASDF data currently only supported with SDF decoder"
            train_dataset = asdf_dataset.ASDFDataset(
                arti_cfg.split_name,
                subsample_amount=1e12,  # Very big
                train=True,
                load_ram=True,
                load_gt=True,
            )
            test_dataset = asdf_dataset.ASDFDataset(
                arti_cfg.split_name,
                subsample_amount=1e12,  # Very big
                train=False,
                load_ram=True,
                load_gt=True,
            )

        if "train_dataset" in additional_outputs.keys():
            additional_outputs["train_dataset"] = train_dataset
        if "test_dataset" in additional_outputs.keys():
            additional_outputs["test_dataset"] = test_dataset

    print(arti_cfg.dataset)
    # Create ShapeEmbedding
    if arti_cfg.dataset == config.DatasetChoice.ours:
        category_list = np.array(
            [
                PartNetMobilityV0DB.get_object_meta(object_id)["model_cat"]
                for object_id in object_index.keys()
            ]
        )
    else:
        category_list = np.array(
            ["_".join(object_id.split("_")[:-1]) for object_id in object_index.keys()]
        )

    shape_embedding = embedding.ShapeEmbedding(
        object_embedding_parameters["weight"], object_index, category_list=category_list
    )

    # Create JointEmbedding class for joint_state_dict --> articulation code!
    joint_config_amount = len(joint_config_index)
    object_amount = len(object_index)

    # These lists should align with the order of joint_config_parameters["weight"]
    # TODO Nick maybe it makes more sense to also save these dicts?
    zero_joint_config_dicts = load_dict(local_dir, "zero_joint_configs")
    joint_definition_dicts = load_dict(local_dir, "joint_definition_dicts")
    object_ids = load_dict(local_dir, "object_ids")
    if not zero_joint_config_dicts or not joint_definition_dicts or not object_ids:
        zero_joint_config_dicts = [None] * joint_config_amount
        joint_definition_dicts = [None] * joint_config_amount
        object_ids = [None] * joint_config_amount
        train_datapoint: dataset.DataPoint
        for train_datapoint in tqdm.tqdm(train_dataset):
            idx = joint_config_index[str(train_datapoint.joint_config_id)]
            zero_joint_config_dicts[idx] = train_datapoint.zero_joint_config
            joint_definition_dicts[idx] = train_datapoint.joint_def
            object_ids[idx] = train_datapoint.object_id

        print("Saving Dict Lists")
        save_dict(zero_joint_config_dicts, local_dir, "zero_joint_configs")
        save_dict(joint_definition_dicts, local_dir, "joint_definition_dicts")
        save_dict(object_ids, local_dir, "object_ids")

    # Make sure object_ids align with joint_ids
    for joint_id, joint_idx in full_index.items():
        joint_object_id = "_".join(joint_id.split("_")[:-1])

        assert (
            joint_object_id == object_ids[joint_idx]
        ), "Misaligned object id to joint id"

    joint_embedding = embedding.JointEmbedding(
        joint_config_parameters["weight"],
        zero_joint_config_dicts,
        joint_definition_dicts,
        object_ids,
    )

    if arti_cfg.scene_decoder == config.SceneDecoder.SDF:
        decoder = SDF_decoder.Decoder(
            arti_cfg.sdf_model_config,
            object_latent_code_dim=arti_cfg.shape_embedding_size,
            joint_config_latent_code_dim=arti_cfg.articulation_embedding_size,
        )
    else:
        assert False, "Unknown Decoder Encountered"

    decoder.load_state_dict(model_params)
    if to_cuda:
        decoder.cuda()
    decoder.eval()

    joint_state_decoder_ = joint_state_decoder.JointStateDecoder(
        arti_cfg.joint_state_decoder_config, arti_cfg.articulation_embedding_size
    )
    joint_state_decoder_.load_state_dict(joint_state_model_params)
    if to_cuda:
        joint_state_decoder_.cuda()
    joint_state_decoder_.eval()

    return (
        decoder,
        joint_state_decoder_,
        shape_embedding,
        joint_embedding,
        additional_outputs,
    )


def get_svd_projectors(X: np.ndarray, dim=2):
    X_mean = X.mean(axis=0)
    U, S, Vh = np.linalg.svd(X - X_mean)
    print(f"{S = }")

    def project_to_lower(X, dim_=dim):
        return (X - X_mean) @ Vh.T[:, :dim_]
        # return X @ Vh[:dim, :].T

    def project_to_higher(X, dim_=dim):
        return (X @ Vh.T[:, :dim_].T) + X_mean
        # return X @ Vh[:dim, :]

    return project_to_lower, project_to_higher


def get_svd_projectors_torch(X: torch.Tensor, dim=2):
    X_mean = X.mean(dim=0)
    U, S, Vh = torch.svd(X - X_mean)
    print(f"{S = }")

    X_mean = X_mean.cuda()
    Vh = Vh.cuda()

    def project_to_lower(X, dim_=dim):
        return (X - X_mean) @ Vh.T[:, :dim_]
        # return X @ Vh[:dim, :].T

    def project_to_higher(X, dim_=dim):
        return (X @ Vh.T[:, :dim_].T) + X_mean
        # return X @ Vh[:dim, :]

    return project_to_lower, project_to_higher


JOINT_TYPE_MAPPING = {"revolute": 0, "prismatic": 1}


def get_joint_type(jt_idx):
    for joint_type, idx in JOINT_TYPE_MAPPING.items():
        if idx == jt_idx:
            return joint_type


def get_joint_type_batch(input_vec):
    _, max_idx = input_vec.max(dim=-1)
    joint_types = [get_joint_type(idx) for idx in max_idx]
    return joint_types


def encode_joint_types(input_types, one_hot: bool = False):
    if isinstance(input_types, str):
        input_types = [input_types]
    type_idx_list = torch.LongTensor(
        [JOINT_TYPE_MAPPING[in_type] for in_type in input_types]
    )
    if not one_hot:
        return type_idx_list
    return torch.nn.functional.one_hot(
        type_idx_list, num_classes=len(JOINT_TYPE_MAPPING)
    )


def extract_type_and_value(joint_definitions, joint_configs):
    """
    Assumes joint_config is already zero!
    """
    joint_types = []
    joint_values = []
    for joint_config, joint_def in zip(joint_configs, joint_definitions):
        joint_id = list(joint_config.keys())[0]
        joint_types.append(joint_def[joint_id]["type"])
        joint_values.append(joint_config[joint_id])
        # gt_types = [list(joint_dict.values())[0]["type"] for joint_dict in gt_joint_config.keys()]
    return joint_types, joint_values


def extract_zero_one_in_limits(joint_definitions, joint_configs):
    """
    Assumes joint_config is already zero!
    """
    zero_one_vals = []
    for joint_config, joint_def in zip(joint_configs, joint_definitions):
        joint_id = list(joint_config.keys())[0]
        joint_value = joint_config[joint_id]
        limits = joint_def[joint_id]["limit"]
        range = limits[1] - limits[0]
        assert joint_value >= 0 - 1e-5, f"{joint_value} >= {0}"
        assert joint_value <= range + 1e-5, f"{joint_value} <= {range}"
        zero_one_vals.append(joint_value / range)
    return zero_one_vals


def get_joint_def_with_canonical_limits(object_id, prismatic_scale: float = 1.0):
    object_meta = PartNetMobilityV0DB.get_object_meta(object_id)
    # This might update directly in-memory
    # TODO Copy?
    joint_def_dict = copy.deepcopy(object_meta["joints"])
    for joint_id in joint_def_dict.keys():
        joint_def_dict[joint_id]["limit"] = partnet_mobility.get_canonical_joint_limits(
            object_meta, joint_id
        )
        if joint_def_dict[joint_id]["type"] == "prismatic":
            joint_def_dict[joint_id]["limit"] = (
                np.array(joint_def_dict[joint_id]["limit"]) / prismatic_scale
            )
    return joint_def_dict


def copy_dict_keys(dict_in, keys):
    return {key: dict_in[key] for key in keys}


def remove_dict_keys(dict_in, keys):
    """
    Removes the keys from the input dict and returns them in a new dict
    """
    return dict_in, {key: dict_in.pop(key) for key in keys}


def subsample_points(mask, sdf, points, subsample):
    ind = np.random.choice(np.count_nonzero(mask), size=subsample)
    # print(f"{ind.shape = } {ind.max()} {ind.min()}")
    return sdf[mask][ind], points[mask][ind]


def equal_sdf_split(
    sdf_values, points, n_points, split_method="smaller", threshold=0.0
):
    """
    Returns a subset of all points, equaly split between pos points (sdf <= 0, inside the object) and neg points (sdf >0, outside the object)
    """
    if split_method == "abs":
        mask = sdf_values.abs() <= threshold
    elif split_method == "smaller":
        mask = sdf_values <= threshold
    else:
        raise Exception(f"Unknown {split_method = }")
    # Positive samples, i.e. inside the object
    sdf_pos, points_pos = subsample_points(mask, sdf_values, points, n_points // 2)
    # Negative samples, i.e. outside of the object
    sdf_neg, points_neg = subsample_points(~mask, sdf_values, points, n_points // 2)
    return np.concatenate([sdf_pos, sdf_neg]), np.concatenate([points_pos, points_neg])


def expanded_coordinates(length: float = 1.0, dim: int = 3, positive_only: bool = True):
    possible_coords = [length, 0.0]
    if not positive_only:
        possible_coords.append(-length)
    coords = list(itertools.product(possible_coords, repeat=dim))
    return coords
