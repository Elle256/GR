import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)  # Lam random lap lap giong nhau
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False