from pickletools import optimize
from torch import optim
import tqdm
import torch
import numpy as np
from CARTO.Decoder import config
from CARTO.Decoder import utils
from CARTO.Decoder.models import lr_schedules
from CARTO.Decoder.embedding import JointEmbedding, ShapeEmbedding


def sdf_to_code(
    decoder: torch.nn.Module,
    points: np.ndarray,
    sdf: np.ndarray,
    args: config.ExperimentConfig,
    sdf_to_code: config.SDFToCodeConfig,
    joint_embedding: JointEmbedding,
    shape_embedding: ShapeEmbedding,
    save_all: bool = False,
    print_progress: bool = False,
    sample_arti_codes_from_cluster_means: bool = True,
):
    """
    Does backward optimization:
      Given a SDF recovers the shape + articulation code
    Assumes data is already correctly chunked
    """
    decoder.eval()
    decoder.cuda()

    assert points.ndim == 2

    N, _ = points.size()
    B = sdf_to_code.samples

    shape_means = torch.ones((B, args.shape_embedding_size)) * torch.Tensor(
        shape_embedding.mean()
    )

    # TODO The whole projection thing could be hidden by a flag!
    joint_low_dim = 16
    project_to_low, project_to_high = utils.get_svd_projectors_torch(
        joint_embedding.embedding_matrix.cuda(), dim=joint_low_dim
    )
    low_dim_joint_embedding = project_to_low(joint_embedding.embedding_matrix.cuda())

    shape_code_samples = torch.normal(
        mean=shape_means, std=sdf_to_code.shape_variance
    ).cuda()
    if sample_arti_codes_from_cluster_means:
        if not sdf_to_code.sample_arti_codes_with_variance:
            arti_code_samples = torch.cat(
                [
                    (
                        project_to_low(torch.Tensor(poly_fit.get_domain_mean()).cuda())
                    ).repeat(repeats=(B // len(joint_embedding.poly_fits), 1))
                    for poly_fit in joint_embedding.poly_fits.values()
                ]
            )
        else:
            # Sample around mean
            arti_code_samples = torch.cat(
                [
                    torch.normal(
                        mean=(
                            project_to_low(
                                torch.Tensor(poly_fit.get_domain_mean()).cuda()
                            )
                        ).repeat(repeats=(B // len(joint_embedding.poly_fits), 1)),
                        std=sdf_to_code.joint_variance,
                    )
                    for poly_fit in joint_embedding.poly_fits.values()
                ]
            )
    else:
        low_dim_mean = project_to_low(joint_embedding.mean().cuda())
        arti_means = torch.ones((B, joint_low_dim)).cuda() * low_dim_mean
        arti_code_samples = torch.normal(
            mean=arti_means, std=sdf_to_code.joint_variance
        ).cuda()

    # GT Code
    # arti_code_samples = (
    #     torch.ones((B, joint_low_dim)).cuda() *
    #     project_to_low(torch.FloatTensor(joint_embedding.poly_fits["revolute"](0.0)).cuda())
    # )

    shape_code = torch.autograd.Variable(shape_code_samples, requires_grad=True)
    arti_code = torch.autograd.Variable(arti_code_samples, requires_grad=True)

    lr_schedule = lr_schedules.LearningRateSchedule.get_from_config(
        sdf_to_code.learning_rate_schedule
    )
    lr_schedule_joint = lr_schedules.LearningRateSchedule.get_from_config(
        sdf_to_code.learning_rate_schedule_joint
    )

    optimizer = torch.optim.Adam(
        [
            {
                # Learning for shape codes
                "params": [shape_code],
                "lr": lr_schedule.get_learning_rate(0),
                "scheduler": lr_schedule,
            },
            {
                # Learning for joint codes
                "params": [arti_code],
                "lr": lr_schedule_joint.get_learning_rate(0),
                "scheduler": lr_schedule_joint,
            },
        ]
    )

    # Combined
    # optimizer = torch.optim.Adam([{
    #     "params": [arti_code, shape_code],
    #     "lr": lr_schedule.get_learning_rate(0),
    #     "scheduler": lr_schedule
    # }])

    loss_l1 = torch.nn.L1Loss(reduction="none")

    total_steps = (
        sdf_to_code.steps + sdf_to_code.only_shape_steps + sdf_to_code.only_joint_steps
    )
    progress = tqdm.tqdm(
        total=total_steps,
        bar_format="{desc:<1.8}{percentage:3.0f}%|{bar:10}{r_bar}",
        leave=False,
    )

    best_loss = np.inf
    best_shape_code = None
    best_arti_code = None
    best_idx = None

    all_losses = []
    all_arti_codes = [project_to_high(arti_code.detach().clone()).cpu()]
    all_shape_codes = [shape_code.detach().clone().cpu()]

    for step in range(total_steps):
        n_points = min(
            args.sdf_model_config.chunk_size,
            points.shape[0],
            args.sdf_model_config.subsample_sdf,
        )
        n_points = 8000

        # Sample equally spaced points
        sdf_t, points_t = utils.equal_sdf_split(sdf, points, n_points)

        sdf_t = torch.Tensor(sdf_t).unsqueeze(-1)
        points_t = torch.Tensor(points_t)

        points_t = points_t.repeat(B, 1)
        sdf_t = sdf_t.repeat(B, 1)

        query_points = torch.split(
            points_t.view(-1, 3), args.sdf_model_config.chunk_size
        )
        sdf_values_gt = torch.split(sdf_t.view(-1, 1), args.sdf_model_config.chunk_size)

        full_losses = torch.zeros((B), requires_grad=False)

        for i in range(len(query_points)):
            stacked_shape_code = utils.stack_ordered(shape_code, n_points)
            stacked_arti_code = utils.stack_ordered(arti_code, n_points)

            network_input = {
                "object_codes": stacked_shape_code.cuda(),
                "joint_config_codes": project_to_high(stacked_arti_code.cuda()),
                # "joint_config_codes": stacked_arti_code.cuda(),
                "points": query_points[i].cuda(),
            }
            sdf_values_pred = decoder(network_input)
            sdf_values_gt_chunk = sdf_values_gt[i].cuda()

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

            # num_query_points, _ = query_points[i].size()
            # print(num_query_points)

            chunk_loss = loss_l1(sdf_values_pred, sdf_values_gt_chunk)
            torch.mean(
                chunk_loss
            ).backward()  # mean --> same scale independent of points/codes

            # Disentangle
            code_losses = torch.reshape(chunk_loss, (B, n_points))

            code_losses = torch.mean(
                code_losses, dim=1
            ).cpu()  # Take the mean along all points

            full_losses += code_losses  # TODO This should be changed when taking the loop multiple times

        reg_shape = sdf_to_code.shape_regularization_norm * torch.mean(
            torch.norm(shape_code, dim=-1)
        )
        reg_shape.backward()

        # dist = torch.abs(arti_code[:, None, :].cuda() - joint_embedding.embedding_matrix.cuda())
        # mean_dist = torch.mean(dist, dim=-1)
        mean_dist = torch.norm(
            arti_code[:, None, :].cuda() - low_dim_joint_embedding, dim=-1
        )
        min_dist = torch.min(mean_dist, dim=1)[0]
        reg_arti = sdf_to_code.joint_regularization_norm * torch.mean(min_dist)
        reg_arti.backward()

        optimizer.step()
        optimizer.zero_grad()

        # # If we are at the last step, delete the shape code from the paramgroup
        if step == sdf_to_code.steps - 1 and sdf_to_code.only_shape_steps > 0:
            # del optimizer.param_groups[0]  # Shape
            del optimizer.param_groups[1]  # Joint
        if (
            step == sdf_to_code.steps + sdf_to_code.only_shape_steps - 1
            and sdf_to_code.only_joint_steps > 0
        ):
            del optimizer.param_groups[0]  # Delete shapes
            if sdf_to_code.only_shape_steps == 0:
                del optimizer.param_groups[0]  # Also delete joints

            optimizer.add_param_group(
                {
                    "params": arti_code,
                    "lr": lr_schedule_joint.get_learning_rate(step),
                    "scheduler": lr_schedule_joint,
                }
            )

        utils.adjust_learning_rate(optimizer, step)

        epoch_best_loss_val, epoch_best_loss_idx = torch.min(full_losses, dim=0)
        epoch_best_loss_val = float(epoch_best_loss_val)

        if epoch_best_loss_val < best_loss:  # or True:
            if print_progress:
                print(
                    f"Decrease from {best_loss = } to {epoch_best_loss_val} at {step = }"
                )
            best_loss = float(epoch_best_loss_val)

            best_shape_code = shape_code[epoch_best_loss_idx, :].detach().clone().cpu()
            best_arti_code = project_to_high(
                arti_code[epoch_best_loss_idx, :].detach().clone()
            ).cpu()
            best_idx = epoch_best_loss_idx

        progress.update(1)
        # progress.set_description_str(f"Best: {best_loss:.8f} Current: {epoch_best_loss_val:.8f}")

        if save_all:
            all_losses.append(full_losses.flatten().detach().clone().cpu())
            all_shape_codes.append(shape_code.detach().clone().cpu())
            all_arti_codes.append(project_to_high(arti_code.detach().clone()).cpu())

        # break

    #### Run on all points
    points_batch = points.repeat(B, 1).cpu()
    shape_code_batch = utils.stack_ordered(shape_code.cpu(), points.size()[0])
    arti_code_batch = utils.stack_ordered(arti_code.cpu(), points.size()[0])

    sdf_values_pred = decoder.safe_forward(
        {
            "points": points_batch,
            "object_codes": shape_code_batch,
            "joint_config_codes": project_to_high(arti_code_batch.cuda()),
        },
        chunk_size=1e5,
    )
    sdf_values_gt = torch.Tensor(sdf).unsqueeze(-1).repeat(B, 1).cpu()

    if args.sdf_model_config.clamp_sdf:
        sdf_values_pred = utils.leaky_clamp(
            sdf_values_pred,
            args.sdf_model_config.clamping_distance,
            args.sdf_model_config.clamping_slope,
        )
        sdf_values_gt = utils.leaky_clamp(
            sdf_values_gt,
            args.sdf_model_config.clamping_distance,
            args.sdf_model_config.clamping_slope,
        )

    all_sdf_loss = loss_l1(sdf_values_pred, sdf_values_gt)
    all_sdf_loss = all_sdf_loss.reshape(B, points.size()[0])

    best_idx_old = best_idx
    summed_losses = torch.sum(all_sdf_loss, dim=1)
    best_idx = int(torch.argmin(summed_losses))
    best_shape_code = shape_code[best_idx, :].detach().clone().cpu()
    best_arti_code = project_to_high(arti_code[best_idx, :].detach().clone()).cpu()

    if best_idx_old != best_idx:
        print(
            f"Found better idx {summed_losses[best_idx_old]} --> {summed_losses[best_idx]}"
        )

    return (
        best_shape_code,
        best_arti_code,
        best_idx,
        all_losses,
        all_arti_codes,
        all_shape_codes,
    )
