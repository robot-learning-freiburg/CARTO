from unicodedata import category
from networkx.algorithms.tree.operations import join
import numpy as np
from sklearn import cluster
import torch
import random
from typing import Dict

from CARTO.Decoder import utils, loss, config
from CARTO.Decoder.multi_poly import MultiPoly
from CARTO.simnet.lib import color_stuff
from CARTO.lib.partnet_mobility import PartNetMobilityV0DB

# from CARTO.simnet.lib import tsne
from openTSNE import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

JOINT_TO_LABEL = {"revolute": "Angle\n[rad]", "prismatic": "Translation\n[m]"}


class EmbeddedSpace:
    def __init__(self, embedding_matrix: torch.Tensor):
        self.embedding_matrix = embedding_matrix.cpu()
        self.proj_to_low, self.proj_to_high = utils.get_svd_projectors(
            self.embedding_matrix.numpy()
        )
        self.projected = self.proj_to_low(self.embedding_matrix.numpy())

    def get_scatter(
        self, ax, additional_points=None, additional_colors=[], points_2d=None
    ):
        mean_projection = self.get_low_mean()
        # ax.scatter(mean_projection[0], mean_projection[1], label="Mean")

        additional_points = np.array(additional_points)
        if additional_points.shape[0] > 0:
            low_dim_points = self.proj_to_low(additional_points)
            ax.scatter(
                low_dim_points[:, 0],
                low_dim_points[:, 1],
                label="Prediction",
                color=additional_colors,
                marker="x",
            )

        if points_2d is None:
            points_2d = self.projected

        # axis_limit_pos = 0.3
        # axis_limit_neg = -0.3
        axis_limit_pos = points_2d.max() * 1.1
        axis_limit_neg = points_2d.min() * 1.1
        # ax.set_xlim(axis_limit_neg, axis_limit_pos)
        # ax.set_ylim(axis_limit_neg, axis_limit_pos)
        # ax.set_aspect('equal', adjustable='box')

        # ax.legend(ncol=3)
        # ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        return ax

    def mean(self):
        return self.embedding_matrix.mean(dim=0)

    def get_low_mean(self):
        mean_code = self.mean()
        return self.proj_to_low(mean_code)

    @property
    def latent_dimensions(self):
        return self.embedding_matrix.size()[1]


class ShapeEmbedding(EmbeddedSpace):
    def __init__(
        self, embedding_matrix: torch.Tensor, object_index, category_list=None
    ):
        super().__init__(embedding_matrix)
        self.object_index = object_index
        self.category_list = category_list

    def get_scatter(
        self,
        ax,
        additional_points=np.empty((0, 2)),
        additional_colors=[],
        mark_means=False,
        do_tsne=False,
    ):
        print("Scatter for ShapeEmbedding")

        unique_categories = set(self.category_list)
        # category_colors = color_stuff.get_colors(len(unique_categories))
        category_colors = sns.color_palette("tab10")[: len(unique_categories)]

        if not do_tsne:
            points_2d = self.projected
        else:
            # points_2d = tsne.tsne(self.embedding_matrix.numpy(), 2, 50, 30.0)
            tsne = TSNE(
                perplexity=30,
                metric="euclidean",
                # n_jobs=8,
                random_state=42,
                verbose=False,
            )
            points_2d = tsne.fit(self.embedding_matrix.numpy())

        for cat, cat_color in zip(unique_categories, category_colors):
            ax.scatter(
                points_2d[self.category_list == cat, 0],
                points_2d[self.category_list == cat, 1],
                label=cat,
                color=cat_color,
            )
            if not mark_means:
                continue

            if not do_tsne:
                mean_2d = np.mean(points_2d[self.category_list == cat], axis=0)
            else:
                # print(self.embedding_matrix.numpy()[self.category_list == cat, :].mean(axis=0))
                mean_high = self.embedding_matrix.numpy()[
                    self.category_list == cat, :
                ].mean(axis=0, keepdims=True)
                # print(mean_high)
                mean_2d = points_2d.transform(mean_high)[0]

            ax.scatter(
                mean_2d[0],
                mean_2d[1],
                # label=cat,
                color=cat_color,
                marker="X",
                s=150,
                edgecolor="black",
                linewidth=1.5,
            )
        ax.set_title("Shape Code Space")
        super().get_scatter(
            ax,
            additional_points=additional_points,
            additional_colors=additional_colors,
            points_2d=points_2d,
        )

    def get_shape_code(self, object_id):
        return self.embedding_matrix[self.object_index[object_id], :]

    def get_category_means(self):
        unique_categories = set(self.category_list)
        category_means = {}
        for cat in unique_categories:
            category_means[cat] = self.embedding_matrix[
                self.category_list == cat, :
            ].mean(dim=0)
        return category_means

    def get_category_shape_code(self, cat, index=-1):
        category_codes = self.embedding_matrix[self.category_list == cat, :]
        if index == -1:
            N = category_codes.size()[0]
            index = random.randint(0, N - 1)
        return category_codes[index, :]


class JointEmbedding(EmbeddedSpace):
    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        joint_config_dicts,
        joint_definitions,
        object_ids,
        K: int = 2,
    ):
        """
        `joint_config_dicts`, `joint_definitions` and `object_ids` should be ordered consistently
        """
        super().__init__(embedding_matrix)
        self.joint_config_dicts = joint_config_dicts
        self.joint_definitions = joint_definitions
        self.K = K
        self.object_ids = object_ids
        self.poly_fits: Dict[str, MultiPoly] = {}

        types, values = utils.extract_type_and_value(
            self.joint_definitions, self.joint_config_dicts
        )
        types = np.array(types)
        values = np.array(values)
        for joint_type in set(types):
            mask = types == joint_type
            embedding_vecs = self.embedding_matrix[mask].numpy()
            self.poly_fits[joint_type] = MultiPoly(
                values[mask], embedding_vecs, poly_dim=5
            )

        for joint_config_dict in joint_config_dicts:
            for joint_val in joint_config_dict.values():
                assert joint_val >= 0

    def get_scatter(
        self,
        ax,
        additional_points=np.empty((0, 2)),
        additional_colors=[],
        group_by_clusters=True,
        category=None,
        dataset_choice: config.DatasetChoice = config.DatasetChoice.ours,
    ):
        # TODO Implement based on clustering
        if group_by_clusters:
            types, values = utils.extract_type_and_value(
                self.joint_definitions, self.joint_config_dicts
            )
            # values = utils.extract_zero_one_in_limits(self.joint_definitions, self.joint_config_dicts)
            types = np.array(types)
            values = np.array(values)

            if category:
                if dataset_choice == config.DatasetChoice.ours:
                    category_list = [
                        PartNetMobilityV0DB.get_object_meta(object_id)["model_cat"]
                        for object_id in self.object_ids
                    ]
                elif dataset_choice == config.DatasetChoice.asdf:
                    category_list = [
                        object_id.split("_")[0] for object_id in self.object_ids
                    ]
                category_list = np.array(category_list)
                category_mask = category_list == category
            else:
                category_mask = np.ones_like(types, dtype=bool)

            markers = ["v", "P", "d"]
            for type, marker in zip(set(types), markers):
                mask = types == type
                mask = np.logical_and(category_mask, mask)
                jt_vals = values[mask]

                # if type == "revolute":
                #   jt_vals *= 180 / np.pi

                color_scatter = ax.scatter(
                    self.projected[mask, 0],
                    self.projected[mask, 1],
                    label=JOINT_TO_LABEL[type],
                    marker=marker,
                    c=jt_vals,
                    cmap="jet",
                    # markeredgewidth=1.5,
                    # markeredgecolor=(1, 1, 1, 1)
                    edgecolor=(0, 0, 0, 1),
                    linewidth=0.5,
                    rasterized=True,
                )
                clb = ax.figure.colorbar(color_scatter)
                # clb.ax.set_title(JOINT_TO_LABEL[type], rotation=-45)
                clb.ax.set_title(JOINT_TO_LABEL[type], rotation=0, fontsize=10)

            # norm = plt.Normalize(np.min(values), np.max(values))
            # sm = plt.cm.ScalarMappable(cmap="jet", norm=norm)
            # sm.set_array([])

        else:
            ax.scatter(self.projected[:, 0], self.projected[:, 1])

        ax.set_title("Joint Code Space")
        super().get_scatter(
            ax, additional_points=additional_points, additional_colors=additional_colors
        )

    def move_along_articulation_line(n_samples=10):
        pass

    def get_joint_type_clusters(self):
        types, values = utils.extract_type_and_value(
            self.joint_definitions, self.joint_config_dicts
        )
        types = np.array(types)
        cluster_points = {}
        for type in set(types):
            cluster_points[type] = self.embedding_matrix[types == type, :]
        return cluster_points

    def get_joint_type_means(self):
        cluster_points = self.get_joint_type_clusters()
        return {cp_id: cps.mean(dim=0) for cp_id, cps in cluster_points.items()}

    def joint_state_to_latent_code(
        self, joint_config, joint_definition, method="NN"
    ) -> torch.Tensor:
        if method == "NN":
            sim_vector = torch.Tensor(
                [
                    loss.articulation_similarity(
                        joint_config_i, joint_def_i, joint_config, joint_definition
                    )
                    for joint_config_i, joint_def_i in zip(
                        self.joint_config_dicts, self.joint_definitions
                    )
                ]
            )
            print(f"{sim_vector.size() = }")
            print(f"{sim_vector[sim_vector.argsort(descending=True)] = }")
            closest_indices = sim_vector.argsort(descending=True)[: self.K]
            # Assumes same ordering of embedding matrix and joint config dicts!
            closest_vectors = self.embedding_matrix[closest_indices, :]
            print(self.joint_config_dicts[closest_indices[0]])
            print(self.joint_config_dicts[closest_indices[1]])
            sim_vectors = sim_vector[closest_indices].unsqueeze(-1)
            # print(sim_vectors)
            # print(closest_vectors)
            # Weighted average
            return (
                (torch.sum(closest_vectors * sim_vectors, dim=0) / sim_vectors.sum())
                .cpu()
                .numpy()
            )
        elif method == "direct":
            assert len(joint_config) == 1
            joint_id = list(joint_config.keys())[0]
            joint_type = joint_definition[joint_id]["type"]
            return self.poly_fits[joint_type](joint_config[joint_id])
        else:
            assert False

    def get_all_poly_plots(self, markers=["v", "P", "d"], n_samples=50):
        all_poly_fits: Dict[str, MultiPoly] = self.poly_fits
        plt_dim = int(np.ceil(np.sqrt(self.latent_dimensions)))
        x_dim = 5
        fig, axes = plt.subplots(
            plt_dim, plt_dim, figsize=(7, 7), sharex=True, sharey=True, dpi=400
        )

        types, values = utils.extract_type_and_value(
            self.joint_definitions, self.joint_config_dicts
        )
        # values = utils.extract_zero_one_in_limits(self.joint_definitions, self.joint_config_dicts)
        types = np.array(types)
        values = np.array(values)

        embedding_matrix = self.embedding_matrix.numpy()

        for i in range(self.latent_dimensions):
            ax = axes[i // plt_dim][i % plt_dim]
            for type, marker in zip(set(types), markers):
                mask = np.array(types) == type
                color_scatter = ax.scatter(
                    values[mask],
                    embedding_matrix[mask, i],
                    label=type,
                    marker=marker,
                    c=values[mask],
                    cmap="jet",
                    edgecolor=(0, 0, 0, 1),
                    linewidth=0.2,
                    rasterized=True,  # Crashes otherwise
                )

            for j, multi_poly in enumerate(all_poly_fits.values()):
                poly = multi_poly.poly_fits[i]
                xx, yy = poly.linspace(n_samples, domain=multi_poly.domain)
                ax.plot(xx, yy, color=sns.color_palette("tab10")[j], linewidth=3)

            # if i == 0:
            #   clb = ax.figure.colorbar(color_scatter)
            #   # clb.ax.set_title(JOINT_TO_LABEL[type], rotation=-45)
            #   clb.ax.set_title(JOINT_TO_LABEL[type], rotation=0, fontsize=10)

        fontsize = 15
        fig.supxlabel(f"Angle [rad]\nTranslation [m]", fontsize=fontsize)
        fig.supylabel("Latent Dimensions", fontsize=fontsize)
        fig.tight_layout()
        fig.legend(["Revolute", "Prismatic"])

        return fig, axes

    def latent_code_to_joint_state(self, latent_code: torch.Tensor):
        pass
