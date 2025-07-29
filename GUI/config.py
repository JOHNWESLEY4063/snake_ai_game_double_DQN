# ddqn_gui/config.py
import torch
import os
from datetime import datetime

# --- General Project Settings ---
PROJECT_NAME = "Snake_DDQN_GUI_AI" # Specific name for this version
MODEL_DIR = './model_weights' # Specific directory for this version
LOG_DIR = './runs' # Specific directory for this version

# Ensure model and log directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Set device for PyTorch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"GUI DDQN Using device: {DEVICE}")

# --- Agent and Training Hyperparameters ---
class AgentConfig:
    MAX_MEMORY = 100_000
    BATCH_SIZE = 1000
    LR = 0.0005
    GAMMA = 0.9
    EPSILON_START = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY_RATE = 0.995
    UPDATE_TARGET_EVERY_GAMES = 100
    SAVE_CHECKPOINT_EVERY_GAMES = 500

    # File names for saving/loading (specific to this GUI version)
    MAIN_MODEL_FILE = 'snake_q_model_gui.pth'
    TARGET_MODEL_FILE = 'snake_target_q_model_gui.pth'
    REPLAY_MEMORY_FILE = 'replay_memory_gui.pkl'
    TRAINING_STATE_FILE = 'training_state_gui.pkl'

# --- Network Architecture ---
class NetworkConfig:
    INPUT_SIZE = 8 + 4 + 4 # Total 16 features
    HIDDEN_SIZE_1 = 256
    HIDDEN_SIZE_2 = 128
    OUTPUT_SIZE = 3

# --- Game Settings ---
class GameConfig:
    WIDTH = 640
    HEIGHT = 480
    BLOCK_SIZE = 20
    SPEED = 40

    REWARD_FOOD = 10
    REWARD_COLLISION = -10
    REWARD_STEP = -0.01
    REWARD_CLOSER = 0.05
    REWARD_FURTHER = -0.05

    COLLISION_TIMEOUT_MULTIPLIER = 150

    ENABLE_OBSTACLES = False # Default, will be overridden by GUI selection
    NUM_STATIC_OBSTACLES = 5
    STATIC_OBSTACLE_LENGTH_MIN = 2
    STATIC_OBSTACLE_LENGTH_MAX = 5
