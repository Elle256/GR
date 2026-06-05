import numpy as np
from collections import defaultdict
from typing import Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# IID partition
# ─────────────────────────────────────────────────────────────────────────────

def iid_partition(
    dataset,
    num_clients: int
) -> Dict[int, List[int]]:

    """
    Split dataset equally and randomly across clients.
    """

    x, y = dataset

    num_items = len(x) // num_clients

    all_idxs = list(range(len(x)))

    np.random.shuffle(all_idxs)

    client_idxs = {}

    for i in range(num_clients):

        client_idxs[i] = all_idxs[
            i * num_items:
            (i + 1) * num_items
        ]

    return client_idxs


# ─────────────────────────────────────────────────────────────────────────────
# Dirichlet Non-IID partition
# ─────────────────────────────────────────────────────────────────────────────

def dirichlet_partition(
    dataset,
    num_clients: int,
    alpha: float,
    min_samples: int = 10,
) -> Dict[int, List[int]]:

    """
    Non-IID partition using Dirichlet distribution.
    """

    x, y = dataset

    targets = np.array(y)

    classes = np.unique(targets)

    # group indices by class
    class_idxs = {}

    for c in classes:

        class_idxs[int(c)] = np.where(
            targets == c
        )[0]

        np.random.shuffle(class_idxs[int(c)])

    client_idxs = defaultdict(list)

    # split each class across clients
    for c in classes:

        idxs = class_idxs[int(c)]

        # sample proportions
        proportions = np.random.dirichlet(
            np.repeat(alpha, num_clients)
        )

        # convert to counts
        proportions = (
            proportions * len(idxs)
        ).astype(int)

        # fix rounding
        diff = len(idxs) - proportions.sum()

        proportions[np.argmax(proportions)] += diff

        start = 0

        for client_id, count in enumerate(proportions):

            client_idxs[client_id].extend(
                idxs[start:start + count].tolist()
            )

            start += count

    # shuffle each client data
    for client_id in client_idxs:

        np.random.shuffle(
            client_idxs[client_id]
        )

    # ensure minimum samples
    all_extra = []

    for client_id in range(num_clients):

        if len(client_idxs[client_id]) < min_samples:

            all_extra.extend(
                client_idxs[client_id]
            )

            client_idxs[client_id] = []

    # redistribute extras
    extra_idx = 0

    for client_id in range(num_clients):

        while (
            len(client_idxs[client_id]) < min_samples
            and extra_idx < len(all_extra)
        ):

            client_idxs[client_id].append(
                all_extra[extra_idx]
            )

            extra_idx += 1

    return dict(client_idxs)


# ─────────────────────────────────────────────────────────────────────────────
# Unified API
# ─────────────────────────────────────────────────────────────────────────────

def partition_dataset(
    dataset,
    num_clients: int,
    iid: bool = False,
    alpha: float = 0.5,
) -> Dict[int, List[int]]:

    """
    Unified interface.
    """

    if iid:

        return iid_partition(
            dataset,
            num_clients
        )

    else:

        return dirichlet_partition(
            dataset,
            num_clients,
            alpha
        )


# ─────────────────────────────────────────────────────────────────────────────
# Label distribution diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def compute_label_distribution(
    dataset,
    client_idxs: Dict[int, List[int]],
) -> Dict[int, Dict[int, int]]:

    """
    Compute per-client label distribution.
    """

    x, y = dataset

    targets = np.array(y)

    dist = {}

    for cid, idxs in client_idxs.items():

        labels, counts = np.unique(
            targets[idxs],
            return_counts=True
        )

        dist[cid] = dict(
            zip(
                labels.tolist(),
                counts.tolist()
            )
        )

    return dist