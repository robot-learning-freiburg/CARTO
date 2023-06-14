import dataclasses
from typing import List, Optional, Dict, Any, Union
import enum
import datetime
import copy
import pathlib
import torch
from CARTO.Decoder.models import lipschitz_norm

import uuid

# Assumes structure like this
# CARTO/
#     CARTO/
#         Decoder/
#             config.py
#     datasets/
#         decoder/
BASE_DIR = (
    pathlib.Path(__file__).parent.resolve()
    / ".."
    / ".."
    / pathlib.Path("datasets/decoder/")
)


def dfac_cur_time():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def default_field(obj):
    return dataclasses.field(default_factory=lambda: copy.copy(obj))


class LearningRateScheduleType(enum.Enum):
    STEP = enum.auto()
    WARMUP = enum.auto()
    CONSTANT = enum.auto()
    LEVEL_DECAY = enum.auto()


class WeightNormalizer(enum.Enum):
    NONE = enum.auto()
    DEFAULT = enum.auto()
    SPECTRAL = enum.auto()
    LEARNED_LIPSCHITZ = enum.auto()


def get_weight_normalizer(normalizer: WeightNormalizer):
    if normalizer == WeightNormalizer.NONE:
        return lambda x: x
    elif normalizer == WeightNormalizer.DEFAULT:
        return torch.nn.utils.weight_norm
    elif normalizer == WeightNormalizer.SPECTRAL:
        return torch.nn.utils.spectral_norm
    elif normalizer == WeightNormalizer.LEARNED_LIPSCHITZ:
        return lipschitz_norm.lipschitz_norm
    assert False, "Unknown weight normalizer requested"


class SceneDecoder(enum.Enum):
    SDF = enum.auto()
    # Could add more here, e.g. NERFs, ROAD etc.


class JointDecoderOutputHeadStyle(enum.Enum):
    CLASSIFICATION = enum.auto()
    ZERO_ONE_HEAD = enum.auto()


@dataclasses.dataclass
class LearningRateScheduleConfig:
    type: LearningRateScheduleType = LearningRateScheduleType.CONSTANT
    initial: float = 1e-4
    interval: Optional[float] = None
    factor: Optional[float] = None
    final: Optional[float] = None
    length: Optional[float] = None


@dataclasses.dataclass
class SDFToCodeConfig:
    cur_time: str = dataclasses.field(default_factory=dfac_cur_time)
    steps: int = 400
    learning_rate_schedule: LearningRateScheduleConfig = default_field(
        LearningRateScheduleConfig(
            type=LearningRateScheduleType.STEP, initial=5e-2, interval=100, factor=0.5
        )
    )
    learning_rate_schedule_joint: LearningRateScheduleConfig = default_field(
        LearningRateScheduleConfig(
            type=LearningRateScheduleType.STEP, initial=1e-2, interval=100, factor=0.5
        )
    )
    samples: int = 16
    shape_variance: float = 5e-1
    joint_variance: float = 3e-1
    shape_regularization_norm: float = 5e-3
    joint_regularization_norm: float = 1e-2
    only_shape_steps: int = 100
    only_joint_steps: int = 100
    subsample: int = 8000
    additional_joint_code_steps: int = -1  #### Legacy
    sample_arti_codes_with_variance: bool = False


@dataclasses.dataclass
class SDFModelConfig:
    # dims: List[int] = default_field([512, 512, 512, 512, 512, 512, 512, 512])
    dims: List[int] = default_field([512, 512, 512, 512])  # Simple
    # dropout: List[int] = default_field([0, 1, 2, 3, 4, 5, 6, 7])
    dropout: List[int] = default_field([])  # Simple
    dropout_prob: float = 0.2
    # latent_dropout: bool = True
    latent_dropout: bool = False  # Simple
    norm_layers: List[int] = default_field([0, 1, 2, 3, 4, 5, 6, 7])
    # joint_config_in: List[int] = default_field([0, 3])
    joint_config_in: List[int] = default_field([0, 3])  # Simple
    # Where feed in the coordinate to render (>=1!)
    xyz_point_in: List[int] = default_field([1, 4])
    weight_normalizer: WeightNormalizer = WeightNormalizer.DEFAULT
    # For a single 16GB GPU, 500000 seem to be max?
    chunk_size: int = 400000
    lipschitz_loss_scale: float = 1e-6  # From original paper
    lipschitz_learning_rate: float = 1e-4  # From original paper

    clamp_sdf: bool = True
    clamping_distance: float = 0.1
    clamping_slope: float = 0.001
    subsample_sdf: int = 12500  # Fits exactly batch size of 32 in chunk size of 400,000, 400,000/32= 12500
    cache_in_ram: bool = True

    sdf_to_code: SDFToCodeConfig = default_field(SDFToCodeConfig())


@dataclasses.dataclass
class JointStateDecoderModelConfig:
    dims: List[int] = default_field([64])  # Simple
    dropout_prob: float = 0.2
    output_head: JointDecoderOutputHeadStyle = (
        JointDecoderOutputHeadStyle.CLASSIFICATION
    )
    weight_normalizer: WeightNormalizer = WeightNormalizer.DEFAULT


class JointCodeSimEnforcement(enum.Enum):
    epoch = enum.auto()
    batch = enum.auto()
    no_enforcement = enum.auto()


class DatasetChoice(enum.Enum):
    ours = enum.auto()
    asdf = enum.auto()


@dataclasses.dataclass
class ExperimentConfig:
    sdf_model_config: SDFModelConfig = default_field(SDFModelConfig())
    joint_state_decoder_config: JointStateDecoderModelConfig = default_field(
        JointStateDecoderModelConfig()
    )
    scene_decoder: SceneDecoder = SceneDecoder.SDF

    local_experiment_id: str = dataclasses.field(
        default_factory=lambda: str(uuid.uuid4())
    )
    pretrained_weights: Optional[str] = None

    dataset: DatasetChoice = DatasetChoice.ours

    # Wouldn't recommend higher
    batch_size: int = 32
    epochs: int = 5000
    # shape_embedding_size: int = 128
    shape_embedding_size: int = 32  # Simple
    articulation_embedding_size: int = 16
    training_data_dir: str = "datasets/decoder/generated_data/All_Real_Categories_50"
    split_name: Union[str, List[str]] = ""  # default_field(["splits/Category/"])
    split_file_name: str = "ids.yaml"
    cur_time: str = dataclasses.field(default_factory=dfac_cur_time)
    regularize_joint_config_code_norm: bool = True
    joint_config_code_regularizer: float = 1e-3
    regularize_shape_config_code_norm: bool = True
    shape_config_regularizer: float = 1e-4
    enforce_joint_config_sim: JointCodeSimEnforcement = JointCodeSimEnforcement.epoch
    joint_config_sim_multiplier: float = 1e-2

    log_imgs_frequency: int = 20
    log_embedding_frequency: int = 10
    log_parameters_frequency: int = 10
    log_val_frequency: int = 10

    hard_sampling: bool = False
    pretrain_joint_config_embedding_steps: int = 5000
    pretrain_joint_config_batch_size: int = 128
    seed: int = 123
    num_workers: int = 8

    eval_metric: str = "chamfer"

    joint_state_loss_multi: float = 1e-1
    joint_state_loss_state_multi: float = 1.0
    joint_state_loss_class_multi: float = 1e-2

    network_parameters_learning_schedule: LearningRateScheduleConfig = default_field(
        LearningRateScheduleConfig(
            type=LearningRateScheduleType.STEP, initial=0.001, interval=400, factor=0.5
        )
    )

    joint_state_decoder_learning_schedule: LearningRateScheduleConfig = default_field(
        LearningRateScheduleConfig(
            type=LearningRateScheduleType.STEP, initial=0.0001, interval=400, factor=0.5
        )
    )

    embedding_parameters_learning_schedule: LearningRateScheduleConfig = default_field(
        LearningRateScheduleConfig(
            type=LearningRateScheduleType.STEP, initial=0.001, interval=400, factor=0.5
        )
    )

    def to_dict(self, expand: bool = True):
        return transform_dict(dataclasses.asdict(self), expand)


#### Specific base configs
descriptions = {}
base_configs = {}
# SDF
descriptions["sdf"] = "Train using SDF as decoder"
base_configs["sdf"] = ExperimentConfig(scene_decoder=SceneDecoder.SDF)
# Could add more decoders here


### Generation Related ###
class ObjectOrigin(enum.Enum):
    PARTNETMOBILITY = enum.auto()
    BASE = enum.auto()
    DEFAULT_STATE = enum.auto()
    CLOSED_STATE = enum.auto()


@dataclasses.dataclass
class GenerationConfig:
    # Prefix for the dataset name
    prefix: str = ""
    # Suffix for the dataset name
    suffix: str = ""
    # PartNetMobility Cateogories to use
    categories: List[str] = default_field([""])
    # Overwrite all filterings steps and use this specific file id
    id_file: Optional[str] = None
    # Number of samples for the SDF
    sdf_number_samples: int = 100000
    # Number of surface points
    pc_number_samples: int = 131072
    # Amount of configurations each object is put in
    num_configs: int = 50
    # Parallalizes across multiple cores
    parallel: bool = True
    # Amount of workers used
    max_workers: int = 40
    # Threshold of when we consider a prismatic joint as moveable
    min_prismatic: float = 0.1
    # Threshold of when we consider a revolute joint as moveable
    min_revolute: float = 0.1
    # 1 parent is always "world" (or whatever the name is) and the other is the base part
    max_unique_parents: int = 2
    # Maximum number of joints, proceed with caution for > 1
    max_joints: int = 1
    # Whether an unlimited joint is ok
    no_limit_ok: bool = False
    # Allowed joint types for all movable
    allowed_joints: List[str] = default_field(["prismatic", "revolute"])
    # Origin frame in which we canonicalize the objects in
    origin_frame: ObjectOrigin = ObjectOrigin.CLOSED_STATE

    # Will be filled while data is being generated
    max_extent: float = 0.0

    # exclude_categories: Dict[str, List[str]] = default_field(...)

    def to_dict(self, expand: bool = True):
        return transform_dict(dataclasses.asdict(self), expand)


def transform_dict(config_dict: Dict, expand: bool = True):
    """
    General function to transform any dictionary into wandb config acceptable format
    (This is mostly due to datatypes that are not able to fit into YAML format which makes wandb angry)
    The expand argument is used to expand iterables into dictionaries so that these configs can be used when compare across runs
    """
    ret: Dict[str, Any] = {}
    for k, v in config_dict.items():
        if v is None or isinstance(v, (int, float, str)):
            ret[k] = v
        elif isinstance(v, (list, tuple, set)):
            # Need to check if item in iterable is YAML-friendly
            t = transform_dict(dict(enumerate(v)), expand)
            # Transform back to iterable if expand is False
            ret[k] = t if expand else [t[i] for i in range(len(v))]
        elif isinstance(v, dict):
            ret[k] = transform_dict(v, expand)
        elif isinstance(v, enum.Enum):
            ret[k] = v.__str__()
        else:
            # Transform to YAML-friendly (str) format
            # Need to handle both Classes, Callables, Object Instances
            # Custom Classes might not have great __repr__ so __name__ might be better in these cases
            vname = v.__name__ if hasattr(v, "__name__") else v.__class__.__name__
            ret[k] = f"{v.__module__}:{vname}"
    return ret
