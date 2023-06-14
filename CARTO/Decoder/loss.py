from collections import defaultdict
from typing import Dict, List, Union, Any
import itertools
import numpy as np

import urdfpy
import torch
from CARTO.Decoder import utils


def articulation_similarity(
    A: Dict[str, float],
    A_def: Dict[str, Any],
    B: Dict[str, float],
    B_def: Dict[str, Any],
    max_values: Dict[str, float] = defaultdict(lambda: 1),
) -> float:
    """
    Calculates the similarity between two two-level kinematic trees including their joint state
    Make sure the joint states are in a canonical state!
    """
    # TODO Nick: For now it's very simple
    # --> Only one joint
    assert len(A) == 1 and len(B) == 1

    joint_id_A = list(A.keys())[0]
    joint_id_B = list(B.keys())[0]

    sim: float
    if A_def[joint_id_A]["type"] == B_def[joint_id_B]["type"]:
        joint_state_A = A[joint_id_A]
        joint_state_B = B[joint_id_B]
        max_joint_state = max_values[A_def[joint_id_A]["type"]]
        # print(max_joint_state)
        # L1
        # dist = np.abs(joint_state_A - joint_state_B)
        # L2
        dist = ((joint_state_A - joint_state_B) / max_joint_state) ** 2
        sim = utils.exp_kernel(dist)
        # sim = utils.distance_to_sim(dist)
        # sim = utils.gauss_kernel(dist)
    else:
        sim = 0.0

    return sim


def get_articulation_similarity_matrix(
    joint_configs: List[Dict[str, float]], joint_definitions: List[Dict[str, Any]]
):
    """
    Returns a matrix of size NxN given a list of N joint config dicts
    """
    # Get max values
    # max_values = utils.AccumulatorDict(accumulator=max)
    # for joint_config, joint_def in zip(joint_configs, joint_definitions):
    #   joint_id = list(joint_config.keys())[0]
    #   max_values.increment(joint_def[joint_id]["type"], joint_config[joint_id])
    max_values = {"prismatic": 0.5, "revolute": 3 / 2 * np.pi}

    sim_matrix = torch.tensor(
        [
            [
                articulation_similarity(
                    joint_config_i,
                    joint_def_i,
                    joint_config_j,
                    joint_def_j,
                    max_values=max_values,
                )
                for joint_config_i, joint_def_i in zip(joint_configs, joint_definitions)
            ]
            for joint_config_j, joint_def_j in zip(joint_configs, joint_definitions)
        ]
    )
    return sim_matrix


class JointSimLoss(torch.nn.Module):
    def __init__(self, joint_config_sim_matrix):
        super(JointSimLoss, self).__init__()
        self.joint_config_sim_matrix = joint_config_sim_matrix

    def forward(self, embedding_matrix):
        """
        Calculate the distance loss
        """
        # Different sim/distance metrics
        # https://elar.urfu.ru/bitstream/10995/3713/2/RuSSIR_2011_07.pdf
        # http://dep805.ru/about/sologub/russir2011poster.pdf
        # embedding_sim = utils.self_cosine_similarity(joint_config_embedding.weight)
        # Not working great --> Bug?

        # embedding_sim = utils.distance_to_sim(
        #     utils.self_manhattan_distance(joint_config_embedding.weight)
        # )

        # embedding_sim = utils.distance_to_sim(
        #     utils.self_euclidean_distance(joint_config_embedding.weight)
        # )

        # embedding_sim = torch.exp(-utils.self_manhattan_distance(joint_config_embedding.weight))

        embedding_sim = torch.exp(-utils.self_euclidean_distance(embedding_matrix))
        joint_config_embedding_loss = torch.nn.functional.l1_loss(
            embedding_sim, self.joint_config_sim_matrix
        )
        # joint_config_embedding_loss = joint_config_embedding_loss / (embedding_matrix.size()[0]**2)
        return joint_config_embedding_loss


class JointClassificationLoss(torch.nn.Module):
    def __init__(self, multi_class: float = 1.0, multi_state: float = 1.0):
        super(JointClassificationLoss, self).__init__()
        self.multi_class = multi_class
        self.multi_state = multi_state

    def forward(
        self,
        gt_joint_configs: List[Dict[str, float]],
        gt_joint_definitions: List[Dict[str, Any]],
        pred_vector: Dict[str, torch.Tensor],
    ):
        """
        Assumes gt_joint_configs is in zerod state!
        """
        assert len(gt_joint_configs[0]) == 1

        # Extract pred
        pred_types_one_hot: torch.Tensor = pred_vector["type"]
        pred_joint_states: torch.Tensor = pred_vector["state"]

        # Extract GT from batch
        gt_types, joint_values = utils.extract_type_and_value(
            gt_joint_definitions, gt_joint_configs
        )
        gt_types_index = utils.encode_joint_types(gt_types).to(
            pred_types_one_hot.device
        )
        gt_joint_states = (
            torch.Tensor(joint_values).to(pred_joint_states.device).unsqueeze(-1)
        )

        class_loss = torch.nn.functional.cross_entropy(
            pred_types_one_hot, gt_types_index
        )
        state_loss = torch.nn.functional.mse_loss(pred_joint_states, gt_joint_states)

        return self.multi_class * class_loss + self.multi_state * state_loss, {
            "class": class_loss.item(),
            "state": state_loss.item(),
        }


class JointZeroOneLoss(torch.nn.Module):
    def __init__(self):
        super(JointZeroOneLoss, self).__init__()

    def forward(
        self,
        gt_joint_configs: List[Dict[str, float]],
        gt_joint_definitions: List[Dict[str, Any]],
        pred_vector: Dict[str, torch.Tensor],
    ):
        """
        Assumes gt_joint_configs is in zerod state!
        """
        assert len(gt_joint_configs[0]) == 1

        pred_zero_one = pred_vector["state"]
        gt_zero_one = (
            torch.Tensor(
                utils.extract_zero_one_in_limits(gt_joint_definitions, gt_joint_configs)
            )
            .to(pred_zero_one.device)
            .unsqueeze(-1)
        )

        loss = torch.nn.functional.mse_loss(pred_zero_one, gt_zero_one)
        # print(f"{pred_zero_one}\n{gt_zero_one}")
        return loss, {"zero_one_loss": loss.item()}
