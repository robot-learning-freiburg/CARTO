from random import sample
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import logging

import open3d as o3d

from typing import List

from CARTO.Decoder import utils
from CARTO.Decoder import config
from CARTO.Decoder.visualizing import code_vis

try:
    import pytorch3d.loss
except Exception as e:
    logging.critical(e, exc_info=True)  # log exception info at CRITICAL log level


class Decoder(nn.Module):
    def __init__(
        self,
        cfg: config.SDFModelConfig,
        object_latent_code_dim: int = 128,
        joint_config_latent_code_dim: int = 16,
    ):
        super(Decoder, self).__init__()
        first_layer_dim = object_latent_code_dim
        if 0 in cfg.joint_config_in:
            first_layer_dim += joint_config_latent_code_dim
        dims = [first_layer_dim] + cfg.dims + [1]
        # First layer takes the object latent code as input
        # Last layer outputs a single SDF value
        # self.fc1 = nn.utils.weight_norm(nn.Linear(joint_config_latent_code_dim + 3, 256))
        self.num_layers = len(dims)
        self.cfg: config.SDFModelConfig = cfg

        self.lipschitz_normers = []

        for layer in range(0, self.num_layers - 1):
            out_dim = dims[layer + 1]
            if layer + 1 in cfg.joint_config_in:
                out_dim -= joint_config_latent_code_dim
            if layer + 1 in cfg.xyz_point_in:
                out_dim -= 3  # xyz

            linear_layer = nn.Linear(dims[layer], out_dim)

            if layer in cfg.norm_layers:
                weight_normalizer = config.get_weight_normalizer(cfg.weight_normalizer)
                if cfg.weight_normalizer == config.WeightNormalizer.LEARNED_LIPSCHITZ:
                    linear_layer, lipschitz_normer = weight_normalizer(linear_layer)
                    # setattr(self, "lip_norm" + str(layer), lipschitz_normer)
                    self.lipschitz_normers.append(lipschitz_normer)
                else:
                    linear_layer = weight_normalizer(linear_layer)

            setattr(self, "lin" + str(layer), linear_layer)

        self.relu = nn.ReLU()
        self.th = nn.Tanh()
        print(self)

    def forward(self, input):
        # TODO Nick: This uses a lot of memory becasue we repeat all indices --> could this be optimized?
        object_latent_codes = input["object_codes"]  # B * SAMPLES x EMB_DIM
        joint_config_latent_codes = input["joint_config_codes"]  # B * SAMPLES x EMB_DIM
        sample_points = input["points"]  # B * Samples x 3

        if self.cfg.latent_dropout:
            object_latent_codes = F.dropout(
                object_latent_codes, p=0.2, training=self.training
            )
            joint_config_latent_codes = F.dropout(
                joint_config_latent_codes, p=0.2, training=self.training
            )

        x = object_latent_codes

        for layer in range(0, self.num_layers - 1):
            if layer in self.cfg.joint_config_in:
                x = torch.cat([x, joint_config_latent_codes], 1)
            if layer in self.cfg.xyz_point_in:
                x = torch.cat([x, sample_points], 1)

            lin = getattr(self, "lin" + str(layer))
            x = lin(x)
            if layer < self.num_layers - 2:
                # x = torch.relu(x)
                x = self.relu(x)
                if layer in self.cfg.dropout:
                    x = F.dropout(x, p=self.cfg.dropout_prob, training=self.training)

        x = self.th(x)

        return x

    def safe_forward(self, input, chunk_size=1e6, return_normals=False):
        """
        Forward pass for reconstruction with marching cubes
        """
        # Hook definition
        grads = {}

        def save_grad(name):
            def hook(grad):
                grads[name] = grad

            return hook

        chunk_size = int(chunk_size)
        # num_points, _ = input["points"].size()

        splitted_points = torch.split(input["points"], chunk_size)
        splitted_object_codes = torch.split(input["object_codes"], chunk_size)
        splitted_config_codes = torch.split(input["joint_config_codes"], chunk_size)

        sdf_results = []
        normal_results = []
        for i, (points, object_codes, config_codes) in enumerate(
            zip(splitted_points, splitted_object_codes, splitted_config_codes)
        ):
            points = points.cuda()
            object_codes = object_codes.cuda()
            config_codes = config_codes.cuda()
            # Input points
            if return_normals:
                xyz = torch.autograd.Variable(points, requires_grad=True)
                xyz.register_hook(save_grad("grid_points"))
            else:
                xyz = points

            sdf_result = self.forward(
                {
                    "points": xyz,
                    "object_codes": object_codes,
                    "joint_config_codes": config_codes,
                }
            )

            if return_normals:
                # Get Jacobian / normals: backprop to vertices to get a Jacobian
                sdf_result.sum().backward(retain_graph=True)
                normals = grads["grid_points"][:, :]

                normals = F.normalize(normals, dim=-1)
                normal_results.append(normals.detach().cpu())

            sdf_results.append(sdf_result.detach().cpu())

            # Delete from GPU
            del points
            del object_codes
            del config_codes
            del sdf_result

        sdf_results = torch.cat(sdf_results).cpu()
        if not return_normals:
            return sdf_results
        return sdf_results, torch.cat(normal_results).cpu()

    def code_forward(
        self,
        object_code: torch.Tensor,
        config_code: torch.Tensor,
        chunk_size=1e6,
        lod_start=8,
        lod_current=8,
        initial_cube_dim: float = 1.0,
        debug_print: bool = False,
        return_normals: bool = False,
        **kwargs,
    ):
        """
        Given codes returns SDF values
        """
        assert object_code.ndim == 1
        assert config_code.ndim == 1
        # Move to CPU to keep GPU empty
        object_code = object_code.cpu()
        config_code = config_code.cpu()
        lod = lod_start
        flat_voxels = utils.get_flat_voxels(
            N_per_axis=2**lod + 1,
            dim_per_axis=[initial_cube_dim, initial_cube_dim, initial_cube_dim],
        )  # Equal distribution
        if debug_print:
            print(f"Starting at {lod = } with {flat_voxels.shape = }")
        while True:  # At least one iteration
            N = flat_voxels.size()[0]
            object_code_stacked = utils.stack_ordered(object_code.unsqueeze(0), N)
            latent_code_stacked = utils.stack_ordered(config_code.unsqueeze(0), N)
            forward_results = self.safe_forward(
                {
                    "object_codes": object_code_stacked,
                    "joint_config_codes": latent_code_stacked,
                    "points": flat_voxels,
                },
                chunk_size=chunk_size,
                return_normals=return_normals,
                **kwargs,
            )
            if return_normals:
                sdf, normals = forward_results
            else:
                sdf = forward_results

            if lod >= lod_current:
                if debug_print:
                    print(f"Reached {lod_current = }")
                break
            cube_side = (initial_cube_dim * 2) / (2 ** (lod + 1))
            cube_diag = np.sqrt(3) * cube_side
            positions = flat_voxels[torch.abs(sdf[:, 0]) < cube_diag]
            # No object found for reconstruction (this happens in the beginning of the training)
            if positions.shape[0] == 0:
                if debug_print:
                    print("positions empty")
                break
            # positions = flat_voxels
            delta = cube_side
            deltas = torch.Tensor(utils.expanded_coordinates(length=delta, dim=3))
            if debug_print:
                print(
                    f"{positions.shape[0]}/{flat_voxels.shape[0]}: {positions.shape[0]/flat_voxels.shape[0]}"
                )
            flat_voxels = (positions[:, None, :] + deltas).flatten(0, 1)
            in_bounds_mask = torch.abs(flat_voxels) < (1.0 + 1e-5)
            in_bounds_mask = torch.all(in_bounds_mask, dim=1)
            flat_voxels = flat_voxels[in_bounds_mask, :]

            lod += 1

        if not return_normals:
            return flat_voxels, sdf

        return flat_voxels, sdf, normals

    def process_codes(
        self,
        obj_codes: torch.Tensor,
        joint_config_codes: torch.Tensor,
        chunk_size: float = 1e6,
        return_normals: bool = False,
        **kwargs,
    ):
        """
        Returns `[points], [sdfs]`
        """

        obj_codes = obj_codes.cuda()
        joint_config_codes = joint_config_codes.cuda()

        all_points, all_sdfs, all_normals = [], [], []
        # Paralellize with https://pytorch.org/docs/stable/multiprocessing.html
        for obj_code, joint_config_code in zip(obj_codes, joint_config_codes):
            # print(joint_config_code)
            torch.cuda.empty_cache()
            forward_results = self.code_forward(
                obj_code,
                joint_config_code,
                chunk_size=chunk_size,
                return_normals=return_normals,
                **kwargs,
            )
            if return_normals:
                points, sdf, normals = forward_results
                all_normals.append(normals.cpu())
            else:
                points, sdf = forward_results

            all_points.append(points.cpu())
            all_sdfs.append(sdf.cpu())

        if not return_normals:
            return all_points, all_sdfs

        return all_points, all_sdfs, all_normals

    def get_ply_meshes(
        self,
        obj_codes: torch.Tensor,
        joint_config_codes: torch.Tensor,
        chunk_size: float = 1e6,
        distance_threshold: float = 0.0,
        all_points=None,
        all_sdfs=None,
        estimate_normals: bool = False,
        **kwargs,
    ):
        """
        estimate_normals: bool
          Whether to estimate the normals based on the pc
            True: uses o3d.estimate_normals()
            False: calculates the normal from the SDF output
        """
        if all_points is None or all_sdfs is None:
            processed_code_results = self.process_codes(
                torch.FloatTensor(obj_codes),
                torch.FloatTensor(joint_config_codes),
                chunk_size=chunk_size,
                return_normals=not estimate_normals,
                **kwargs,
            )
            if estimate_normals:
                all_points, all_sdfs = processed_code_results
            else:
                all_points, all_sdfs, all_normals = processed_code_results

        ply_objects = []
        for idx, (points, sdf) in enumerate(zip(all_points, all_sdfs)):
            # ply_object = code_vis.convert_sdf_samples_to_ply(points, sdf, threshold=distance_threshold)
            mesh = o3d.geometry.PointCloud()
            sdf_mask = torch.abs(sdf[:, 0]) <= distance_threshold
            # sdf_mask = sdf[:, 0] <= distance_threshold
            if estimate_normals:
                mesh.points = o3d.utility.Vector3dVector(
                    points[sdf_mask].detach().cpu().numpy()
                )
                mesh.estimate_normals()
            else:
                normals = all_normals[idx]
                points = points[sdf_mask] - (sdf[sdf_mask] * normals[sdf_mask])
                mesh.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
                mesh.normals = o3d.utility.Vector3dVector(
                    normals[sdf_mask].detach().cpu().numpy()
                )
            ply_objects.append(mesh)
        return ply_objects

    def get_lipschitz_term_product(self):
        # lipschitz_loss = torch.tensor([1.0], requires_grad=True).to(chunk_loss.device)
        lipschitz_product = 1.0
        for normer in self.lipschitz_normers:
            lipschitz_product *= torch.nn.functional.softplus(normer.lipschitz_constant)
        return lipschitz_product


def prepare_inputs(
    batch,
    object_indices_flat,
    joint_config_indices_flat,
    chunk_size: int = 0,
):
    query_points = batch["points"].cuda()
    sdf_values_gt = batch["sdf"].cuda()
    if not torch.abs(query_points[sdf_values_gt <= 0.0]).max() <= (
        1.0 + 1e-1
    ):  # Allow some error margin due to unclean meshes/random sampling, Background: we calulcated the max extent based on the PC sampling and not SDF --> small marginal errors
        raise AssertionError(
            f"Query points belonging to object out of unit cube with {torch.abs(query_points[sdf_values_gt <= 0.0]).max()}"
        )
    B, num_query_points, _ = query_points.size()
    query_points = query_points.view(-1, 3)
    sdf_values_gt = sdf_values_gt.view(-1, 1)
    query_points.requires_grad = False
    sdf_values_gt.requires_grad = False

    # Stack according to points
    object_indices = utils.stack_ordered(
        object_indices_flat.unsqueeze(-1), num_query_points
    ).cuda()
    joint_config_indices = utils.stack_ordered(
        joint_config_indices_flat.unsqueeze(-1), num_query_points
    ).cuda()

    if chunk_size > 0:
        object_indices = torch.split(object_indices, chunk_size)
        joint_config_indices = torch.split(joint_config_indices, chunk_size)
        query_points = torch.split(query_points, chunk_size)
        sdf_values_gt = torch.split(sdf_values_gt, chunk_size)

    # print(f"{num_query_points = }")
    # print(f"{len(sdf_values_gt)}")
    # for sdf_vals in sdf_values_gt:
    #   print(f"{sdf_vals.size() = }")

    return (
        object_indices,
        joint_config_indices,
        query_points,
        sdf_values_gt,
        num_query_points,
    )


def calculate_sdf_loss(
    decoder: "Decoder",
    loss_fun,
    batch,
    object_indices_flat,
    joint_config_indices_flat,
    object_embedding,
    joint_config_embedding,
    args: config.ExperimentConfig,
):
    B = len(batch["object_id"])
    (
        object_indices,
        joint_config_indices,
        query_points,
        sdf_values_gt,
        num_query_points,
    ) = prepare_inputs(
        batch,
        object_indices_flat,
        joint_config_indices_flat,
        chunk_size=args.sdf_model_config.chunk_size,
    )

    # Split batch in small subbatches
    batch_losses = utils.AccumulatorDict()

    for i in range(len(object_indices)):
        object_codes = object_embedding(object_indices[i].squeeze(-1))
        joint_config_codes = joint_config_embedding(joint_config_indices[i].squeeze(-1))

        network_input = {
            "object_codes": object_codes,
            "joint_config_codes": joint_config_codes,
            "points": query_points[i],
        }

        sdf_values_pred = decoder(network_input)
        sdf_values_gt_chunk = sdf_values_gt[i]

        if args.sdf_model_config.clamp_sdf:
            sdf_values_pred = utils.leaky_clamp(
                sdf_values_pred,
                args.sdf_model_config.clamping_distance,
                args.sdf_model_config.clamping_slope,
            )
            sdf_values_gt_chunk = utils.leaky_clamp(
                sdf_values_gt_chunk,
                args.sdf_model_config.clamping_distance,
                args.sdf_model_config.clamping_slope,
            )

        chunk_loss = loss_fun(sdf_values_pred, sdf_values_gt_chunk) / num_query_points
        batch_losses.increment("loss/batch/reconstruction", chunk_loss.item() / B)

        chunk_loss.backward()
    return batch_losses


def calculate_val_metric(
    decoder: "Decoder",
    batch,
    object_indices_flat,
    joint_config_indices_flat,
    object_embedding,
    joint_config_embedding,
    args: config.ExperimentConfig,
):
    B = len(batch["object_id"])
    (
        object_indices,
        joint_config_indices,
        query_points,
        sdf_values_gt,
        num_query_points,
    ) = prepare_inputs(
        batch,
        object_indices_flat,
        joint_config_indices_flat,
        chunk_size=args.sdf_model_config.chunk_size,
    )
    # Empty tensor for storing sdf results
    all_sdf_values_pred = torch.empty_like(
        torch.cat(sdf_values_gt), requires_grad=False
    )

    # Iterate over all chunks
    for i in range(len(object_indices)):
        object_codes = object_embedding(object_indices[i].squeeze(-1))
        joint_config_codes = joint_config_embedding(joint_config_indices[i].squeeze(-1))

        network_input = {
            "object_codes": object_codes,
            "joint_config_codes": joint_config_codes,
            "points": query_points[i],
        }

        sdf_values_pred = decoder(network_input)
        # Store predicted SDF value
        all_sdf_values_pred[
            i
            * args.sdf_model_config.chunk_size : (i + 1)
            * args.sdf_model_config.chunk_size
        ] = sdf_values_pred.detach()

    # Reshape everything into batch style
    sdf_values_gt = torch.cat(sdf_values_gt).view(B, num_query_points)
    sdf_values_pred = all_sdf_values_pred.view(B, num_query_points)
    query_points = torch.cat(query_points).view(B, num_query_points, 3)

    # Iterate over the batch one by one for chamfer distance
    all_val_metrics = []
    for i in range(B):
        pc_gt = query_points[i][sdf_values_gt[i] < 0.0][None, ...]

        threshold = 0.0
        if torch.count_nonzero(sdf_values_pred[i] < threshold) == 0:
            threshold = sdf_values_pred[i].min() + 1e-5
        pc_pred = query_points[i][sdf_values_pred[i] < threshold][None, ...]

        # TODO Fix Bug on servers!
        chamfer_loss, _ = pytorch3d.loss.chamfer_distance(pc_pred, pc_gt)
        chamfer_loss = chamfer_loss.item() * 1000
        # chamfer_loss = torch.Tensor([1.0])
        all_val_metrics.append(
            {
                "meta_info": {
                    "object_id": str(batch["object_id"][i]),
                    "joint_config_id": str(batch["joint_config_id"][i]),
                },
                "chamfer": chamfer_loss,
                "l1": float(
                    torch.nn.functional.l1_loss(
                        sdf_values_pred[i].cpu(), sdf_values_gt[i].cpu()
                    ).item()
                ),
            }
        )
    return all_val_metrics
