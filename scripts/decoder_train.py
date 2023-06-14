import math
import pathlib
import random

import tyro
import numpy as np
import torch

# Debug stuff
import tqdm
import wandb

# from pytorch3d import loss as pytorch3d_loss
from CARTO.Decoder import loss, utils, config
from CARTO.Decoder.config import (
    BASE_DIR,
    ExperimentConfig,
    WeightNormalizer,
    GenerationConfig,
)
from CARTO.Decoder.data import dataset, asdf_dataset
from CARTO.Decoder.models import SDF_decoder, lr_schedules
from CARTO.Decoder.models.joint_state_decoder import JointStateDecoder
from CARTO.Decoder.visualizing import offscreen

import copy

# torch.autograd.set_detect_anomaly(True)


def main(args: ExperimentConfig):
    # Create run directory etc.
    local_experiment_directory = utils.save_cfg(BASE_DIR, args)

    if args.dataset == config.DatasetChoice.ours:
        gen_cfg: GenerationConfig = utils.load_cfg(
            pathlib.Path(args.training_data_dir), cfg_class=GenerationConfig
        )
        max_extents = gen_cfg.max_extent
        assert max_extents > 0.0, "max_extents incorrect"
        rescaler = dataset.Rescaler3D(scale=max_extents)

        split_dicts = dataset.get_dataset_split_dict(
            pathlib.Path(args.training_data_dir),
            args.split_name,
            file_name=args.split_file_name,
        )
        train_dataset = dataset.SimpleDataset(split_dicts["train"], rescaler=rescaler)
        val_dataset = dataset.SimpleDataset(split_dicts["val"], rescaler=rescaler)
        # test_dataset = dataset.SimpleDataset(split_dicts["test"], rescaler=rescaler)
    elif args.dataset == config.DatasetChoice.asdf:
        assert (
            args.scene_decoder == config.SceneDecoder.SDF
        ), "ASDF data currently only supported with SDF decoder"
        train_dataset = asdf_dataset.ASDFDataset(
            args.split_name,
            subsample_amount=args.sdf_model_config.subsample_sdf,
            load_ram=args.sdf_model_config.cache_in_ram,
            train=True,
        )
        val_dataset = asdf_dataset.ASDFDataset(
            args.split_name,
            subsample_amount=1e12,  # Very big
            load_ram=args.sdf_model_config.cache_in_ram,
            train=True,
        )

    assert len(train_dataset) == len(val_dataset)

    # Create index
    object_index = utils.IndexDict()
    joint_config_index = utils.IndexDict()
    full_index = utils.IndexDict()

    # Make one pass to populate indices
    print("Building object + joint config indices")
    zero_joint_config_dicts = []
    joint_definition_dicts = []

    datapoint: dataset.DataPoint

    for i, datapoint in tqdm.tqdm(enumerate(train_dataset), total=len(train_dataset)):
        object_index[datapoint.object_id]
        joint_config_index[str(datapoint.joint_config_id)]
        full_index[f"{datapoint.object_id}_{str(datapoint.joint_config_id)}"]

        # Get rescaled jointdef/zero_joint_config
        joint_definition_dicts.append(copy.deepcopy(datapoint.joint_def))
        zero_joint_config_dicts.append(copy.deepcopy(datapoint.zero_joint_config))

    utils.save_index(object_index, local_experiment_directory, "object")
    utils.save_index(joint_config_index, local_experiment_directory, "joint_config")
    utils.save_index(full_index, local_experiment_directory, "full")

    custom_sampler = (
        torch.utils.data.sampler.WeightedRandomSampler(
            weights=torch.ones(len(train_dataset)) / len(train_dataset),
            num_samples=len(train_dataset),
            replacement=True,
        )
        if args.hard_sampling
        else None
    )

    if args.dataset == config.DatasetChoice.ours:
        train_dataset = dataset.SDFDataset(
            split_dicts["train"],
            subsample=args.sdf_model_config.subsample_sdf,
            cache_in_ram=args.sdf_model_config.cache_in_ram,
            rescaler=rescaler,
        )
        val_dataset = dataset.SDFDataset(
            split_dicts["val"],
            subsample=100000000,  # Very big value
            cache_in_ram=args.sdf_model_config.cache_in_ram,
            rescaler=rescaler,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=not args.hard_sampling,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn,
        pin_memory=False,  # TODO Interfers with our random subsampling in __getitem__?
        sampler=custom_sampler,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    # TODO Nick: Add normalization here on XYZ (trans + scale) and SDF values (only scale)

    if args.enforce_joint_config_sim == config.JointCodeSimEnforcement.epoch:
        print("Get Articulation Similarity Matrix")
        matrix_path = local_experiment_directory / "sim_matrix.npy"
        if matrix_path.exists():
            joint_config_index_loaded = utils.load_index(
                local_experiment_directory, "joint_config"
            )
            assert (
                joint_config_index == joint_config_index_loaded
            ), "Joint Config Indices do not match"
            joint_config_sim_matrix = np.load(
                local_experiment_directory / "sim_matrix.npy"
            )
            joint_config_sim_matrix = torch.Tensor(joint_config_sim_matrix).cuda()
        else:
            joint_config_sim_matrix = loss.get_articulation_similarity_matrix(
                zero_joint_config_dicts, joint_definition_dicts
            )
            # Write matrix
            np.save(matrix_path, joint_config_sim_matrix.numpy())

    # Create embeddings
    object_embedding = torch.nn.Embedding(
        len(object_index), args.shape_embedding_size
    ).cuda()
    joint_config_embedding = torch.nn.Embedding(
        len(joint_config_index), args.articulation_embedding_size
    ).cuda()

    # Set default values
    torch.nn.init.normal_(
        object_embedding.weight.data,
        0.0,
        1.0 / math.sqrt(args.shape_embedding_size),
    )
    torch.nn.init.normal_(
        joint_config_embedding.weight.data,
        0.0,
        1.0 / math.sqrt(args.articulation_embedding_size),
    )

    print("Loading and generating model")
    if args.scene_decoder == config.SceneDecoder.SDF:
        decoder = SDF_decoder.Decoder(
            args.sdf_model_config,
            object_latent_code_dim=args.shape_embedding_size,
            joint_config_latent_code_dim=args.articulation_embedding_size,
        )
    else:
        assert False, "Unknown Decoder Encountered"
    decoder = decoder.cuda()

    if args.pretrained_weights:
        full_checkpoint = torch.load(args.pretrained_weights)
        decoder.load_state_dict(full_checkpoint["model_parameters"])

    joint_state_decoder = JointStateDecoder(
        args.joint_state_decoder_config,
        joint_config_latent_code_dim=args.articulation_embedding_size,
    ).cuda()
    if (
        args.joint_state_decoder_config.output_head
        == config.JointDecoderOutputHeadStyle.CLASSIFICATION
    ):
        joint_state_loss = loss.JointClassificationLoss(
            multi_class=args.joint_state_loss_class_multi,
            multi_state=args.joint_state_loss_state_multi,
        )
    elif (
        args.joint_state_decoder_config.output_head
        == config.JointDecoderOutputHeadStyle.ZERO_ONE_HEAD
    ):
        joint_state_loss = loss.JointZeroOneLoss()

    loss_l1 = torch.nn.L1Loss(reduction="sum")
    network_schedule = lr_schedules.LearningRateSchedule.get_from_config(
        args.network_parameters_learning_schedule
    )
    joint_state_decoder_schedule = lr_schedules.LearningRateSchedule.get_from_config(
        args.joint_state_decoder_learning_schedule
    )
    embedding_schedule = lr_schedules.LearningRateSchedule.get_from_config(
        args.embedding_parameters_learning_schedule
    )

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": network_schedule.get_learning_rate(0),
                "scheduler": network_schedule,
            },
            {
                "params": joint_state_decoder.parameters(),
                "lr": joint_state_decoder_schedule.get_learning_rate(0),
                "scheduler": joint_state_decoder_schedule,
            },
            {
                "params": object_embedding.parameters(),
                "lr": embedding_schedule.get_learning_rate(0),
                "scheduler": embedding_schedule,
            },
            {
                "params": joint_config_embedding.parameters(),
                "lr": embedding_schedule.get_learning_rate(0),
                "scheduler": embedding_schedule,
            },
        ]
    )

    all_schedules = [network_schedule, embedding_schedule]

    if (
        args.scene_decoder == config.SceneDecoder.SDF
        and args.sdf_model_config.weight_normalizer
        == WeightNormalizer.LEARNED_LIPSCHITZ
    ):
        all_lipschitz_parameters = []
        for lipschitz_normer in decoder.lipschitz_normers:
            all_lipschitz_parameters.extend(lipschitz_normer.parameters())
        optimizer_all.add_param_group(
            {"params": all_lipschitz_parameters, "lr": args.lipschitz_learning_rate}
        )

    wandb.init(project="cit_sdf", config=args.to_dict())
    wandb.watch(decoder, log="all", log_freq=args.log_parameters_frequency)
    wandb.watch(object_embedding, log="all", log_freq=args.log_parameters_frequency)
    wandb.watch(
        joint_config_embedding, log="all", log_freq=args.log_parameters_frequency
    )
    wandb.define_metric("epoch")
    custom_batch_counter = 0
    created_graph = False

    if args.pretrain_joint_config_embedding_steps > 0:
        if args.enforce_joint_config_sim == config.JointCodeSimEnforcement.epoch:
            joint_sim_loss = loss.JointSimLoss(joint_config_sim_matrix.cuda())
            # Pretrain Joint Config Embedding
            print(
                f"Pretraining embedding for {args.pretrain_joint_config_embedding_steps} steps"
            )
            for i in tqdm.tqdm(range(args.pretrain_joint_config_embedding_steps)):
                joint_embedding_sim_loss = joint_sim_loss(joint_config_embedding.weight)
                joint_embedding_norm_loss = torch.mean(
                    torch.norm(joint_config_embedding.weight, dim=1)
                )
                joint_config_embedding_loss = (
                    joint_embedding_sim_loss + 0.1 * joint_embedding_norm_loss
                )
                joint_config_embedding_loss.backward()
                optimizer_all.step()
                optimizer_all.zero_grad()

                # S = torch.svd(joint_config_embedding.weight.data, compute_uv=False)
                log_dict = {
                    "pretrain/step": i,
                    "pretraining/joint_config_embedding_loss": joint_config_embedding_loss.item(),
                    "pretraining/joint_config_embedding_sim": joint_embedding_sim_loss.item(),
                    "pretraining/joint_config_embedding_norm": joint_embedding_norm_loss.item(),
                }
                wandb.log(log_dict)
        if args.enforce_joint_config_sim == config.JointCodeSimEnforcement.batch:
            pretrain_dataset = dataset.SimpleDataset(
                split_dicts["train"], cache_only_meta=True, rescaler=rescaler
            )
            pretrain_loader = torch.utils.data.DataLoader(
                pretrain_dataset,
                shuffle=True,
                batch_size=args.pretrain_joint_config_batch_size,
                collate_fn=pretrain_dataset.collate_meta,
                pin_memory=True,
                num_workers=0,
            )
            # TODO: Weighted sampling on joint type to ensure equal distribution of prismatic/revolute?
            for i in tqdm.tqdm(range(args.pretrain_joint_config_embedding_steps)):
                log_dict = utils.AccumulatorDict({"pretrain/step": i})
                for batch in tqdm.tqdm(pretrain_loader, leave=False):
                    joint_config_indices = [
                        joint_config_index[joint_conf_id]
                        for joint_conf_id in batch["joint_config_id"]
                    ]

                    zjcd_batch = [
                        zero_joint_config_dicts[idx] for idx in joint_config_indices
                    ]
                    jdd_batch = [
                        joint_definition_dicts[idx] for idx in joint_config_indices
                    ]
                    joint_config_sim_matrix = loss.get_articulation_similarity_matrix(
                        zjcd_batch, jdd_batch
                    ).cuda()
                    joint_sim_loss = loss.JointSimLoss(joint_config_sim_matrix)

                    joint_config_codes = joint_config_embedding(
                        torch.IntTensor(joint_config_indices).cuda()
                    )
                    joint_embedding_norm_loss = torch.mean(
                        torch.norm(joint_config_codes, dim=1)
                    )
                    joint_embedding_sim_loss = joint_sim_loss(joint_config_codes)
                    joint_config_embedding_loss = (
                        10
                        * (joint_embedding_sim_loss + 1e-3 * joint_embedding_norm_loss)
                        / len(batch["object_id"])
                    )
                    # Scale by 10 to make gradient bigger

                    joint_config_embedding_loss.backward()
                    optimizer_all.step()
                    optimizer_all.zero_grad()

                    log_dict.increment(
                        "pretraining/joint_config_embedding_loss",
                        joint_config_embedding_loss.item()
                        * len(batch["object_id"])
                        / len(pretrain_dataset),
                    )
                    log_dict.increment(
                        "pretraining/joint_config_embedding_sim",
                        joint_embedding_sim_loss.item()
                        * len(batch["object_id"])
                        / len(pretrain_dataset),
                    )
                    log_dict.increment(
                        "pretraining/joint_config_embedding_norm",
                        joint_embedding_norm_loss.item()
                        * len(batch["object_id"])
                        / len(pretrain_dataset),
                    )

                # Log every 1%-th step
                if i % int(0.01 * args.pretrain_joint_config_embedding_steps):
                    wandb.log(log_dict)
                # Log every 10%-th step
                if i % int(0.1 * args.pretrain_joint_config_embedding_steps):
                    wandb.log(
                        {
                            "pretrain/step": i,
                            "joint_config_embedding": wandb.Table(
                                columns=[
                                    f"D{i}"
                                    for i in range(args.articulation_embedding_size)
                                ],
                                data=joint_config_embedding.weight.cpu()
                                .detach()
                                .numpy(),
                            ),
                        }
                    )

    wandb.log(
        {
            "joint_config_embedding": wandb.Table(
                columns=[f"D{i}" for i in range(args.articulation_embedding_size)],
                data=joint_config_embedding.weight.cpu().detach().numpy(),
            )
        }
    )

    best_metric_value = np.inf

    print(f"Starting main loop for {args.epochs} epochs")
    for epoch in tqdm.tqdm(range(args.epochs)):
        utils.adjust_learning_rate(optimizer_all, epoch)

        epoch_losses = utils.AccumulatorDict()
        decoder.train()
        joint_state_decoder.train()
        for batch in tqdm.tqdm(train_loader, desc="Train Batch", leave=False):
            if batch is None:
                print("Encountered rare case of batch == None")
                continue

            optimizer_all.zero_grad()

            B = len(batch["object_id"])

            object_indices_flat = torch.IntTensor(
                [object_index[obj_id] for obj_id in batch["object_id"]]
            ).cuda()
            joint_config_indices_flat = torch.IntTensor(
                [
                    joint_config_index[joint_conf_id]
                    for joint_conf_id in batch["joint_config_id"]
                ]
            ).cuda()

            # Decoder Losses
            if args.scene_decoder == config.SceneDecoder.SDF:
                batch_losses = SDF_decoder.calculate_sdf_loss(
                    decoder,
                    loss_l1,
                    batch,
                    object_indices_flat,
                    joint_config_indices_flat,
                    object_embedding,
                    joint_config_embedding,
                    args,
                )
            else:
                assert False, "Unknown Decoder Encountered"

            # Optimize joint state
            joint_codes = joint_config_embedding(joint_config_indices_flat)
            joint_state_pred = joint_state_decoder(joint_codes)
            js_loss_calc, js_loss_dict = joint_state_loss(
                batch["zero_joint_config"], batch["joint_definition"], joint_state_pred
            )
            js_loss_calc *= args.joint_state_loss_multi
            js_loss_calc.backward()
            batch_losses["loss/batch/joint_state"] = js_loss_calc.item() / B
            for loss_name, loss_val in js_loss_dict.items():
                batch_losses[f"loss/batch/joint_state/{loss_name}"] = loss_val / B

            # Optional Regularizers
            if args.regularize_shape_config_code_norm:
                object_codes = object_embedding(object_indices_flat)
                shape_embedding_loss = args.shape_config_regularizer * torch.mean(
                    torch.norm(object_codes, dim=1)
                )
                batch_losses[
                    "loss/batch/shape_embedding_reg"
                ] = shape_embedding_loss.item()
            if args.regularize_joint_config_code_norm:
                joint_config_codes = joint_config_embedding(joint_config_indices_flat)
                joint_embedding_loss = args.joint_config_code_regularizer * torch.mean(
                    torch.norm(joint_config_codes, dim=1)
                )
                batch_losses[
                    "loss/batch/joint_embedding_reg"
                ] = joint_embedding_loss.item()

            if args.enforce_joint_config_sim == config.JointCodeSimEnforcement.batch:
                # Just keep the last one, as it applies to the full dataset
                joint_config_indices = [
                    joint_config_index[joint_conf_id]
                    for joint_conf_id in batch["joint_config_id"]
                ]

                zjcd_batch = [
                    zero_joint_config_dicts[idx] for idx in joint_config_indices
                ]
                jdd_batch = [
                    joint_definition_dicts[idx] for idx in joint_config_indices
                ]
                joint_config_sim_matrix = loss.get_articulation_similarity_matrix(
                    zjcd_batch, jdd_batch
                ).cuda()
                joint_sim_loss = loss.JointSimLoss(joint_config_sim_matrix)

                joint_config_codes = joint_config_embedding(joint_config_indices_flat)
                joint_config_embedding_loss = (
                    args.joint_config_sim_multiplier
                    * joint_sim_loss(joint_config_codes)
                )
                joint_config_embedding_loss.backward()
                batch_losses["loss/batch/joint_config_embedding_sim"] = (
                    joint_config_embedding_loss.item() / B
                )

            batch_losses["loss/batch/full_loss"] = np.sum(list(batch_losses.values()))

            # Take a step
            optimizer_all.step()
            optimizer_all.zero_grad()

            log_dict = {"batch": custom_batch_counter, "epoch": epoch}
            log_dict.update(batch_losses)
            wandb.log(log_dict)
            custom_batch_counter += 1

            # if grad_clip is not None:
            # torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            #
            for full_batch_loss_name in batch_losses.keys():
                loss_name = "/".join(full_batch_loss_name.split("/")[2:])
                # print(f"{loss_name = }")
                epoch_losses.increment(
                    f"loss/epoch/{loss_name}",
                    batch_losses[full_batch_loss_name] / len(train_loader),
                )

        # End of batch for-loop
        # Everything below is execuated after all batches

        if args.enforce_joint_config_sim == config.JointCodeSimEnforcement.epoch:
            # Ensure distances stay the same after our epoch step
            joint_config_embedding_loss = (
                args.joint_config_sim_multiplier
                * len(train_loader)
                * joint_sim_loss(joint_config_embedding.weight)
            )
            # joint_config_embedding_loss += args.joint_config_code_regularizer * torch.sum(
            #     torch.norm(joint_config_embedding.weight, dim=1)
            # )
            joint_config_embedding_loss.backward()
            optimizer_all.step()
            optimizer_all.zero_grad()
            epoch_losses[
                "loss/epoch/joint_config_embedding_sim"
            ] = joint_config_embedding_loss.item()

        # Maybe this should be moved into each batch?
        if (
            args.scene_decoder == config.SceneDecoder.SDF
            and args.sdf_model_config.weight_normalizer
            == WeightNormalizer.LEARNED_LIPSCHITZ
        ):
            optimizer_all.zero_grad()
            lipschitz_loss = decoder.get_lipschitz_term_product()
            lipschitz_loss *= args.lipschitz_loss_scale
            epoch_losses["loss/epoch/lipschitz_constant"] = lipschitz_loss.item()
            lipschitz_loss.backward()
            optimizer_all.step()
            optimizer_all.zero_grad()

        # Epoch ends
        optimizer_all.zero_grad()
        utils.save_checkpoint(
            local_experiment_directory,
            decoder,
            joint_state_decoder,
            optimizer_all,
            object_embedding,
            joint_config_embedding,
            epoch,
            current_epoch=False,
            latest=True,
            best=False,
        )

        # Logging
        log_dict = {"epoch": epoch}
        log_dict.update(epoch_losses)
        log_dict["lr/network"] = network_schedule.get_learning_rate(epoch)
        log_dict[
            "lr/joint_state_decoder"
        ] = joint_state_decoder_schedule.get_learning_rate(epoch)
        log_dict["lr/embedding"] = embedding_schedule.get_learning_rate(epoch)

        if (epoch + 1) % args.log_val_frequency == 0:
            decoder.eval()
            torch.cuda.empty_cache()
            all_val_metrics = []
            # Iterate over whole dataset and collect chamfer distances
            for batch in tqdm.tqdm(val_loader, desc="Val Batch", leave=False):
                object_indices_flat = torch.IntTensor(
                    [object_index[obj_id] for obj_id in batch["object_id"]]
                ).cuda()
                joint_config_indices_flat = torch.IntTensor(
                    [
                        joint_config_index[joint_conf_id]
                        for joint_conf_id in batch["joint_config_id"]
                    ]
                ).cuda()
                # full_indices_flat = torch.IntTensor()

                if args.scene_decoder == config.SceneDecoder.SDF:
                    batch_metric = SDF_decoder.calculate_val_metric(
                        decoder,
                        batch,
                        object_indices_flat,
                        joint_config_indices_flat,
                        object_embedding,
                        joint_config_embedding,
                        args,
                    )
                    all_val_metrics.extend(batch_metric)
                else:
                    assert False, "Unknown Decoder Encountered"

            for metric_name in all_val_metrics[0].keys():
                if metric_name == "meta_info":
                    continue
                # Use collected data
                all_vals = np.array([point[metric_name] for point in all_val_metrics])
                nan_vals = np.count_nonzero(np.isnan(all_vals))
                if nan_vals > 0:
                    print(f"Found {nan_vals}/{len(all_vals)} NaN {metric_name} vals")
                # Store the full chamfer value
                mean_vals = np.nanmean(all_vals)
                log_dict[f"val/{metric_name}"] = mean_vals

                if metric_name == args.eval_metric:
                    eval_metric_value = mean_vals

            # Store single values
            cols = [key for key in all_val_metrics[0].keys()]
            table_object = wandb.Table(columns=cols)
            for single_entry in all_val_metrics:
                table_object.add_data(*[single_entry[key] for key in cols])
            log_dict["val/single"] = table_object

            if eval_metric_value < best_metric_value:
                utils.save_checkpoint(
                    local_experiment_directory,
                    decoder,
                    joint_state_decoder,
                    optimizer_all,
                    object_embedding,
                    joint_config_embedding,
                    epoch,
                    current_epoch=False,
                    latest=False,
                    best=True,
                )
                print(
                    f"Saving best checkpoint with {eval_metric_value} compared to {best_metric_value} on {args.eval_metric}"
                )
                best_metric_value = eval_metric_value

            if args.hard_sampling:
                new_sample_weights = torch.ones_like(custom_sampler.weights)
                for single_entry in all_val_metrics:
                    new_sample_weights[
                        full_index[
                            f"{single_entry['meta_info']['object_id']}_{single_entry['meta_info']['joint_config_id']}"
                        ]
                    ] = single_entry[args.eval_metric]
                new_sample_weights = torch.exp(new_sample_weights) / torch.sum(
                    torch.exp(new_sample_weights)
                )
                log_dict["hard_sample_weights"] = wandb.Histogram(
                    new_sample_weights, num_bins=20
                )
                custom_sampler.weights = new_sample_weights

        # Comment in to create a graph
        # if not created_graph:
        #   created_graph = True
        #   g = torchviz.make_dot(
        #       chunk_loss,
        #       params={
        #           **dict(decoder.named_parameters()),
        #           # **dict(object_embedding.named_parameters()),
        #           # **dict(joint_config_embedding.named_parameters())
        #       }
        #   )
        #   g.render(local_experiment_directory / "reconstruction_loss", view=False)
        #   if args.enforce_joint_config_sim:
        #     g = torchviz.make_dot(
        #         joint_config_embedding_loss,
        #         params={
        #             # **dict(decoder.named_parameters()),
        #             # **dict(object_embedding.named_parameters()),
        #             **dict(joint_config_embedding.named_parameters())
        #         }
        #     )
        #     g.render(local_experiment_directory / "embedding_loss", view=False)

        if (epoch + 1) % args.log_embedding_frequency == 0:
            # Custom Implementation
            # log_dict["object_embedding"] = utils.torch_tensor_to_2_dim_wandb(
            #     object_embedding.weight, utils.svd_projection_np
            # )
            # log_dict["joint_config_embedding"] = utils.torch_tensor_to_2_dim_wandb(
            #     joint_config_embedding.weight, tsne
            # )
            log_dict["object_embedding"] = wandb.Table(
                columns=[f"D{i}" for i in range(args.shape_embedding_size)],
                data=object_embedding.weight.cpu().detach().numpy(),
            )
            log_dict["joint_config_embedding"] = wandb.Table(
                columns=[f"D{i}" for i in range(args.articulation_embedding_size)],
                data=joint_config_embedding.weight.cpu().detach().numpy(),
            )

        if (epoch + 1) % args.log_imgs_frequency == 0:
            # Generate image
            shape_code = object_embedding.state_dict()["weight"][0]
            joint_code = joint_config_embedding.state_dict()["weight"][0]

            if args.scene_decoder == config.SceneDecoder.SDF:
                points, sdf = decoder.code_forward(
                    shape_code,
                    joint_code,
                    chunk_size=args.sdf_model_config.chunk_size,
                    lod_start=4,
                    lod_current=8,
                )
                pred_pc = offscreen.get_point_cloud(
                    points, sdf, color=np.array([1.0, 0.0, 0.0])
                )
            else:
                assert False, "Unknown Decoder Encountered"

            meshes = []
            meshes.append(pred_pc)
            # datapoint = train_dataset[0]
            # gt_pc = offscreen.get_point_cloud(
            #     datapoint.points, datapoint.sdf_values, color=np.array([0., 1., 0.])
            # )
            # meshes.append(gt_pc)
            rgb = offscreen.render_offscreen(meshes=meshes)
            log_dict["rendered_object"] = wandb.Image(
                rgb,
                caption="SDF for first object + joint config code\nred: voxelized pred, green: gt",
            )

        # Log epoch to wandb
        wandb.log(log_dict)


if __name__ == "__main__":
    # args = tyro.parse(ExperimentConfig)
    args = tyro.cli(
        tyro.extras.subcommand_type_from_defaults(
            config.base_configs, config.descriptions
        ),
    )

    torch.random.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)
