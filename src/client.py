from typing import List, Tuple

from .update import (
    LocalUpdate,
    LocalUpdateProx,
)


class FLClient:
    """
    Federated Learning client abstraction.

    Each client:
    - owns local dataset indices
    - performs local training
    - returns updated weights
    """

    def __init__(
        self,
        client_id: int,
        dataset,
        idxs: List[int],
        args,
    ):
        self.client_id = client_id

        self.dataset = dataset

        self.idxs = idxs

        self.args = args

        self.num_samples = len(idxs)

    def train(
        self,
        global_model,
    ) -> Tuple[list, float]:

        """
        Standard FedAvg local training.
        """

        trainer = LocalUpdate(
            self.args,
            self.dataset,
            self.idxs,
        )

        weights, loss = trainer.train(
            global_model
        )

        return weights, loss

    def train_prox(
        self,
        global_model,
    ) -> Tuple[list, float]:

        """
        FedProx local training.
        """

        trainer = LocalUpdateProx(
            self.args,
            self.dataset,
            self.idxs,
        )

        weights, loss = trainer.train(
            global_model,
            global_model,
        )

        return weights, loss

    def __len__(self):

        return self.num_samples

    def __repr__(self):

        return (
            f"FLClient("
            f"id={self.client_id}, "
            f"samples={self.num_samples}"
            f")"
        )