import torch
import numpy as np
import random

def set_seed(seed: int):
    """Sets the seed for reproducibility across libraries and configurations, assuming CUDA usage.
    
    Args:
        seed (int): The seed value to use.
    """
    # Set seed for random module
    random.seed(seed)
    
    # Set seed for numpy
    np.random.seed(seed)
    
    # Set seed for torch (CPU)
    torch.manual_seed(seed)
    
    # Set seed for CUDA (GPU operations)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setup
    
    # Enable deterministic behavior in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_num_threads(1)