import random
import numpy as np
import torch
from numpy.random import RandomState, SeedSequence, MT19937

def set_seed(seed: int = 42) -> RandomState:
    """
    Set random seeds for reproducibility for random, numpy, pytorch (CPU/GPU).

    Returns:
        NumPy RandomState object
    """
    random.seed(seed)
    rs = RandomState(MT19937(SeedSequence(seed)))
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return rs
