# GUI/Helper.py
import random
import numpy as np
import torch

def set_seeds(seed_value=42):
    """
    Sets random seeds for reproducibility across different libraries.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seeds set to {seed_value}")

# Matplotlib plot function removed for GUI version to rely solely on TensorBoard.
