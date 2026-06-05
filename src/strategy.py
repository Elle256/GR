import numpy as np
from typing import List

def fedavg_aggregate(
    local_weights: List[List[np.ndarray]],
    client_sizes: List[int],
) -> List[np.ndarray]:

    assert len(local_weights) == len(client_sizes), \
        "Mismatch between weights and client sizes"

    assert len(local_weights) > 0, \
        "No client updates to aggregate"

    total = sum(client_sizes)

    # initialize aggregated weights
    agg_weights = []

    num_layers = len(local_weights[0])

    for layer_idx in range(num_layers):

        # weighted average initialization
        weighted_layer = (
            local_weights[0][layer_idx] *
            (client_sizes[0] / total)
        )

        # aggregate remaining clients
        for client_w, size in zip(
            local_weights[1:],
            client_sizes[1:]
        ):

            scale = size / total

            weighted_layer += (
                client_w[layer_idx] * scale
            )

        agg_weights.append(weighted_layer)

    return agg_weights


def apply_dropout(
    selected_clients: List[int],
    dropout_rate: float,
) -> List[int]:

    if dropout_rate <= 0.0:
        return selected_clients

    survived = [
        c for c in selected_clients
        if np.random.random() > dropout_rate
    ]

    # ensure at least one client survives
    if len(survived) == 0:
        survived = [
            selected_clients[
                np.random.randint(len(selected_clients))
            ]
        ]

    return survived