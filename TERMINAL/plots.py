# terminal/Helper.py
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
# Keep this for display.clear_output if running in interactive env, otherwise it's safe to remove.

# Enable interactive mode for matplotlib for live updates
plt.ion()

# Set style for better visuals
plt.style.use('fivethirtyeight')

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

def plot(scores, mean_scores):
    """
    Live plotting of scores during training using Matplotlib.
    This function is called by train.py to update the plot.
    """
    # Clear the current figure to prevent overlapping plots
    plt.clf()

    # Title and labels
    plt.title("Snake AI Training Progress (Terminal)", fontsize=14)
    plt.xlabel('Number of Games', fontsize=12)
    plt.ylabel('Score', fontsize=12)

    # Plot data
    plt.plot(scores, label="Score", color='tab:blue')
    plt.plot(mean_scores, label="Mean Score", color='tab:orange')

    # Axis limits and grid
    plt.ylim(ymin=0)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Annotate last points
    if len(scores) > 0:
        plt.text(len(scores)-1, scores[-1], f'{scores[-1]}', ha='right', va='bottom', fontsize=10)
    if len(mean_scores) > 0:
        plt.text(len(mean_scores)-1, mean_scores[-1], f'{mean_scores[-1]:.1f}', ha='right', va='top', fontsize=10)

    plt.legend()
    plt.tight_layout()
    plt.show(block=False) # Non-blocking show
    plt.pause(0.1) # Pause briefly to allow plot to update
