"""
Evaluation metrics for Federated Learning experiments.

Includes:
  - Per-round accuracy / loss tracking
  - Client drift (Γ_k = ||w_k - w_global||)
  - Fairness (std-dev of per-client accuracy)
  - Summary statistics
"""

import numpy as np
import tensorflow as tf
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Running tracker
# ─────────────────────────────────────────────────────────────────────────────

class MetricsTracker:
    """
    Accumulate per-round metrics over a full experiment run.
    """

    def __init__(self):

        self.rounds: List[int] = []

        self.train_losses: List[float] = []

        self.test_losses: List[float] = []

        self.test_accs: List[float] = []

        self.client_drifts: List[float] = []

        self.fairness_scores: List[float] = []

        self.active_clients: List[int] = []

    def update(
        self,
        round_idx: int,
        train_loss: float,
        test_loss: float,
        test_acc: float,
        drift: float = 0.0,
        fairness: float = 0.0,
        n_active: int = 0,
    ) -> None:

        self.rounds.append(round_idx)

        self.train_losses.append(train_loss)

        self.test_losses.append(test_loss)

        self.test_accs.append(test_acc)

        self.client_drifts.append(drift)

        self.fairness_scores.append(fairness)

        self.active_clients.append(n_active)

    def to_dict(self) -> dict:

        return {
            "rounds": self.rounds,
            "train_loss": self.train_losses,
            "test_loss": self.test_losses,
            "test_acc": self.test_accs,
            "client_drift": self.client_drifts,
            "fairness": self.fairness_scores,
            "active_clients": self.active_clients,
        }

    def best_accuracy(self) -> float:

        return max(self.test_accs) if self.test_accs else 0.0

    def rounds_to_target(
        self,
        target_acc: float
    ) -> Optional[int]:

        """
        Return first round reaching target accuracy.
        """

        for r, acc in zip(self.rounds, self.test_accs):

            if acc >= target_acc:
                return r

        return None

    def summary(self) -> dict:

        if not self.test_accs:
            return {}

        return {

            "best_acc":
                self.best_accuracy(),

            "final_acc":
                self.test_accs[-1],

            "final_train_loss":
                self.train_losses[-1],

            "final_test_loss":
                self.test_losses[-1],

            "mean_drift":
                float(np.mean(self.client_drifts)),

            "mean_fairness":
                float(np.mean(self.fairness_scores)),

            "rounds_to_80":
                self.rounds_to_target(0.80),

            "rounds_to_85":
                self.rounds_to_target(0.85),

            "rounds_to_90":
                self.rounds_to_target(0.90),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Client drift
# ─────────────────────────────────────────────────────────────────────────────

def compute_client_drift(
    local_weights: List[List[np.ndarray]],
    global_weights: List[np.ndarray],
) -> float:

    if not local_weights:
        return 0.0

    drifts = []

    for client_w in local_weights:

        diff_sq = 0.0

        for w_local, w_global in zip(
            client_w,
            global_weights
        ):

            diff_sq += np.sum(
                np.square(w_local - w_global)
            )

        drift = np.sqrt(diff_sq)

        drifts.append(drift)

    return float(np.mean(drifts))


# ─────────────────────────────────────────────────────────────────────────────
# Fairness
# ─────────────────────────────────────────────────────────────────────────────

def compute_fairness(
    per_client_accs: List[float]
) -> float:

    """
    Fairness = std of per-client accuracy.

    Lower → fairer model.
    """

    if len(per_client_accs) < 2:
        return 0.0

    return float(np.std(per_client_accs))


# ─────────────────────────────────────────────────────────────────────────────
# Weighted average
# ─────────────────────────────────────────────────────────────────────────────

def weighted_average(
    values: List[float],
    weights: List[int],
) -> float:

    """
    Compute weighted average.
    """

    total = sum(weights)

    if total == 0:
        return 0.0

    return float(
        sum(v * w for v, w in zip(values, weights))
        / total
    )