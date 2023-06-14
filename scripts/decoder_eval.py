import dataclasses
import torch
import tqdm
import pytorch3d
import numpy as np
import enum
import uuid
import random

from CARTO.Decoder import utils
from CARTO.Decoder import config
from CARTO.Decoder.data import dataset
from CARTO.Decoder.models import backward
import tyro


# TODO Move?
class DatasetToTest(enum.Enum):
    train = enum.auto()
    test = enum.auto()


@dataclasses.dataclass
class ReconstructionConfig:
    experiment_id: str
    reconstruction_run_id: str = dataclasses.field(
        default_factory=lambda: str(uuid.uuid4())
    )
    sdf_to_code: config.SDFToCodeConfig = config.default_field(config.SDFToCodeConfig())
    distance_threshold: float = 1e-2
    lod_start: int = 7
    lod: int = 7
    variance_explanation: float = 0.9
    dataset_to_test: DatasetToTest = DatasetToTest.test
    seed: int = 123
    reeval: bool = False
    old_reconstruction_run_id: str = ""  # If this is set we will load from this old dir, and save in new reconstruction dir


@dataclasses.dataclass
class CodesEvalResult:
    chamfer: float
    joint_error: float
    joint_type_correct: bool
    pred_joint_state: float
    pred_joint_type: str
    chamfer_mesh: float = -1  # Legacy
    chamfer_proj: float = -1


@dataclasses.dataclass
class ObjectEvalResult:
    object_id: str
    joint_id: str
    shape_code: np.ndarray
    joint_code: np.ndarray
    original_codes: CodesEvalResult
    reeval_joint_code: CodesEvalResult
    reprojected_shape_code: CodesEvalResult


def eval_codes(
    shape_decoder,
    joint_state_decoder,
    shape_code,
    arti_code,
    example: dataset.DataPoint,
):
    shape_code = shape_code.reshape(1, shape_code.size()[-1]).cuda()
    arti_code = arti_code.reshape(1, arti_code.size()[-1]).cuda()

    points, sdf_values, normals = shape_decoder.process_codes(
        shape_code,
        arti_code,
        lod_start=cfg.lod_start,
        lod_current=cfg.lod,
        return_normals=True,
        chunk_size=2e5,
    )
    # List returned
    points = points[0]
    sdf_values = sdf_values[0]
    normals = normals[0]

    sdf_mask = torch.abs(sdf_values[:, 0]) <= cfg.distance_threshold
    sdf = sdf_values[sdf_mask]
    points = points[sdf_mask]
    normals = normals[sdf_mask]

    # No normal projection
    pred_pc = points
    # Normal projection to ise surface
    pred_pc_proj = points - (sdf * normals)

    pred_pc = pred_pc.detach().unsqueeze(0).cuda()
    pred_pc_proj = pred_pc_proj.detach().unsqueeze(0).cuda()

    gt_pc = torch.FloatTensor(example.full_pc).cuda().unsqueeze(0)

    cd, _ = pytorch3d.loss.chamfer_distance(pred_pc, gt_pc)
    cd_proj, _ = pytorch3d.loss.chamfer_distance(pred_pc_proj, gt_pc)

    print(f"{pred_pc.size() = }")
    print(f"{pred_pc_proj.size() = }")
    print(f"{gt_pc.size() = }")

    # This does not work since we have non-even grids
    # zero_level_ply_object = code_vis.convert_sdf_samples_to_ply(points, sdf_values, threshold=0.0)
    # file_string = str("tmp_ply_mesh.ply")
    # zero_level_ply_object.write(file_string)
    # dense_mesh = trimesh.load(file_string)
    # surface_points = trimesh.sample.sample_surface(dense_mesh, 30000)[0]
    # pred_surface_points = torch.FloatTensor(surface_points).cuda().unsqueeze(0)
    # cd_mesh, _ = pytorch3d.loss.chamfer_distance(pred_surface_points, gt_pc)
    cd_mesh = torch.Tensor([0])

    joint_dict_result = joint_state_decoder(arti_code.cuda())

    pred_joint_state = joint_dict_result["state"][0].detach().cpu()
    pred_joint_type = utils.get_joint_type_batch(joint_dict_result["type"])[0]

    gt_joint_state = list(example.zero_joint_config.values())[0]
    gt_joint_type = example.joint_def[list(example.zero_joint_config.keys())[0]]["type"]

    joint_error = torch.abs(pred_joint_state - gt_joint_state).detach().cpu()
    joint_type_correct = gt_joint_type == pred_joint_type

    return CodesEvalResult(
        chamfer=float(cd.detach().cpu()),
        chamfer_mesh=float(cd_mesh.detach().cpu()),
        chamfer_proj=float(cd_proj.detach().cpu()),
        joint_error=float(joint_error),
        joint_type_correct=joint_type_correct,
        pred_joint_state=float(pred_joint_state),
        pred_joint_type=pred_joint_type,
    )


def main(cfg: ReconstructionConfig):
    local_dir = config.BASE_DIR / "runs" / cfg.experiment_id
    eval_dir = local_dir / "eval" / "reconstruction" / cfg.reconstruction_run_id
    eval_dir.mkdir(exist_ok=True, parents=True)

    if cfg.reeval and cfg.old_reconstruction_run_id != "":
        old_eval_dir = (
            local_dir / "eval" / "reconstruction" / cfg.old_reconstruction_run_id
        )
    else:
        old_eval_dir = eval_dir

    with open(eval_dir / "config.yaml", "w") as file:
        file.write(tyro.to_yaml(cfg))

    (
        decoder,
        joint_state_decoder,
        shape_embedding,
        joint_embedding,
        additional_outputs,
    ) = utils.load_full_decoder(
        cfg.experiment_id,
        additional_outputs={"test_dataset": None, "train_dataset": None, "cfg": None},
    )

    X = shape_embedding.embedding_matrix
    X_mean = X.mean(axis=0)
    _, S, _ = np.linalg.svd(X - X_mean)
    percentage = np.cumsum(S) / np.sum(S)
    low_dim = np.argmax(percentage >= cfg.variance_explanation)
    print(f"Variance idx: {low_dim} / {percentage.shape[0]}")

    if cfg.dataset_to_test == DatasetToTest.test:
        test_dataset = additional_outputs["test_dataset"]
    elif cfg.dataset_to_test == DatasetToTest.train:
        test_dataset = additional_outputs["train_dataset"]

    experiment_cfg = additional_outputs["cfg"]

    all_eval_results = []
    example: dataset.DataPoint
    for example in tqdm.tqdm(test_dataset):
        torch.cuda.empty_cache()
        if not cfg.reeval:
            (
                shape_code,
                arti_code,
                best_idx,
                all_losses,
                all_arti_codes,
                all_shape_codes,
            ) = backward.sdf_to_code(
                decoder,
                torch.Tensor(example.points),
                torch.Tensor(example.sdf_values),
                experiment_cfg,
                cfg.sdf_to_code,
                joint_embedding,
                shape_embedding,
                save_all=False,
            )
        else:
            file_path = (
                old_eval_dir
                / f"{example.object_id}_{str(example.joint_config_id)}.yaml"
            )
            with open(file_path, "r") as fh:
                old_eval_results: ObjectEvalResult = tyro.from_yaml(
                    ObjectEvalResult, fh.read()
                )
                shape_code = torch.Tensor(old_eval_results.shape_code)
                arti_code = torch.Tensor(old_eval_results.joint_code)

        original_eval_result = eval_codes(
            decoder, joint_state_decoder, shape_code, arti_code, example
        )

        # Re Eval
        joint_dict_result = joint_state_decoder(arti_code.cuda().unsqueeze(0))
        pred_joint_state = joint_dict_result["state"][0].detach().cpu()
        pred_joint_type = utils.get_joint_type_batch(joint_dict_result["type"])[0]
        arti_code_reeval = torch.Tensor(
            joint_embedding.poly_fits[pred_joint_type](np.array([pred_joint_state]))
        )
        reevaluated_eval_result = eval_codes(
            decoder, joint_state_decoder, shape_code, arti_code_reeval, example
        )

        # Reprojecting
        shape_code_reprojection = torch.Tensor(
            shape_embedding.proj_to_high(
                shape_embedding.proj_to_low(np.array(shape_code), dim_=low_dim),
                dim_=low_dim,
            )
        ).unsqueeze(0)
        reprojection_eval_result = eval_codes(
            decoder, joint_state_decoder, shape_code_reprojection, arti_code, example
        )

        eval_result = ObjectEvalResult(
            object_id=example.object_id,
            joint_id=example.joint_config_id,
            shape_code=np.array(shape_code),
            joint_code=np.array(arti_code),
            original_codes=original_eval_result,
            reeval_joint_code=reevaluated_eval_result,
            reprojected_shape_code=reprojection_eval_result,
        )

        file_path = (
            eval_dir / f"{eval_result.object_id}_{str(eval_result.joint_id)}.yaml"
        )
        with open(file_path, "w") as fh:
            fh.write(tyro.to_yaml(eval_result))

        all_eval_results.append(eval_result)

        ### Print values for default
        eval_result: ObjectEvalResult
        all_cds = np.array(
            [eval_result.original_codes.chamfer for eval_result in all_eval_results]
        )
        all_cds_mesh = np.array(
            [
                eval_result.original_codes.chamfer_mesh
                for eval_result in all_eval_results
            ]
        )
        all_cds_proj = np.array(
            [
                eval_result.original_codes.chamfer_proj
                for eval_result in all_eval_results
            ]
        )
        all_joint_errors = np.array(
            [eval_result.original_codes.joint_error for eval_result in all_eval_results]
        )
        all_joint_types = np.array(
            [
                eval_result.original_codes.joint_type_correct
                for eval_result in all_eval_results
            ]
        )
        print(f"Chamfer Distance: {(all_cds.mean() * 1000):.3f}")
        print(f"Chamfer Distance (Mesh): {(all_cds_mesh.mean() * 1000):.3f}")
        print(f"Chamfer Distance (Proj): {(all_cds_proj.mean() * 1000):.3f}")
        print(
            f"Joint State Error: {(all_joint_errors.mean()):.5f} rad, {(all_joint_errors.mean() * 180 / np.pi):.3f} deg"
        )
        print(
            f"Joint Type Acc: {(np.count_nonzero(all_joint_types) / len(all_eval_results)):.3f}"
        )


if __name__ == "__main__":
    cfg: ReconstructionConfig = tyro.parse(ReconstructionConfig)

    torch.random.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    main(cfg)
