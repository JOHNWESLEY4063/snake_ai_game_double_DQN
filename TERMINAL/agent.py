# terminal/agent.py
import torch
import random
import numpy as np
from collections import deque
import pickle
import os

# Corrected import: Import AgentConfig class, DEVICE, and MODEL_DIR directly.
# File names are accessed via AgentConfig.
from model import Linear_QNet, QTrainer
from config import AgentConfig, NetworkConfig, DEVICE, MODEL_DIR

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = AgentConfig.EPSILON_START
        self.gamma = AgentConfig.GAMMA
        self.memory = deque(maxlen=AgentConfig.MAX_MEMORY)

        self.model = Linear_QNet(NetworkConfig.INPUT_SIZE,
                                 NetworkConfig.HIDDEN_SIZE_1,
                                 NetworkConfig.HIDDEN_SIZE_2,
                                 NetworkConfig.OUTPUT_SIZE)
        self.target_model = Linear_QNet(NetworkConfig.INPUT_SIZE,
                                        NetworkConfig.HIDDEN_SIZE_1,
                                        NetworkConfig.HIDDEN_SIZE_2,
                                        NetworkConfig.OUTPUT_SIZE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.trainer = QTrainer(self.model, lr=AgentConfig.LR, gamma=self.gamma)

        self.record_score = 0
        self.plot_scores = []
        self.plot_mean_scores = []
        self.total_score = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > AgentConfig.BATCH_SIZE:
            mini_sample = random.sample(self.memory, AgentConfig.BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones, target_model=self.target_model)
        return loss

    def train_short_memory(self, state, action, reward, next_state, done):
        loss = self.trainer.train_step(state, action, reward, next_state, done, target_model=self.target_model)
        return loss

    def get_action(self, state):
        self.epsilon = max(AgentConfig.EPSILON_MIN, AgentConfig.EPSILON_START * (AgentConfig.EPSILON_DECAY_RATE ** self.n_games))
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(DEVICE)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        print("Target network updated.")

    def save_checkpoint(self, file_prefix="checkpoint"):
        # Access file names via AgentConfig
        self.model.save(file_name=f"{file_prefix}_{AgentConfig.MAIN_MODEL_FILE}", model_dir=MODEL_DIR)
        self.target_model.save(file_name=f"{file_prefix}_{AgentConfig.TARGET_MODEL_FILE}", model_dir=MODEL_DIR)

        optimizer_path = os.path.join(MODEL_DIR, f"{file_prefix}_optimizer.pth")
        torch.save(self.trainer.optimizer.state_dict(), optimizer_path)
        print(f"Optimizer state saved to {optimizer_path}")

        memory_path = os.path.join(MODEL_DIR, AgentConfig.REPLAY_MEMORY_FILE)
        with open(memory_path, 'wb') as f:
            pickle.dump(self.memory, f)
        print(f"Replay memory saved to {memory_path}")

        training_state = {
            'n_games': self.n_games,
            'epsilon': self.epsilon,
            'record_score': self.record_score,
            'total_score': self.total_score,
            'plot_scores': self.plot_scores,
            'plot_mean_scores': self.plot_mean_scores
        }
        state_path = os.path.join(MODEL_DIR, AgentConfig.TRAINING_STATE_FILE)
        with open(state_path, 'wb') as f:
            pickle.dump(training_state, f)
        print(f"Training state saved to {state_path}")

    def load_checkpoint(self, file_prefix="checkpoint"):
        # Access file names via AgentConfig
        main_model_path = os.path.join(MODEL_DIR, f"{file_prefix}_{AgentConfig.MAIN_MODEL_FILE}")
        target_model_path = os.path.join(MODEL_DIR, f"{file_prefix}_{AgentConfig.TARGET_MODEL_FILE}")
        optimizer_path = os.path.join(MODEL_DIR, f"{file_prefix}_optimizer.pth")
        memory_path = os.path.join(MODEL_DIR, AgentConfig.REPLAY_MEMORY_FILE)
        state_path = os.path.join(MODEL_DIR, AgentConfig.TRAINING_STATE_FILE)

        if os.path.exists(main_model_path):
            self.model.load_state_dict(torch.load(main_model_path, map_location=DEVICE))
            self.model.eval()
            print(f"Main model loaded from {main_model_path}")
        else:
            print(f"Main model checkpoint not found at {main_model_path}")
            return False

        if os.path.exists(target_model_path):
            self.target_model.load_state_dict(torch.load(target_model_path, map_location=DEVICE))
            self.target_model.eval()
            print(f"Target model loaded from {target_model_path}")
        else:
            print(f"Target model checkpoint not found at {target_model_path}")
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
            print("Target model re-initialized from main model.")

        if os.path.exists(optimizer_path):
            self.trainer.optimizer.load_state_dict(torch.load(optimizer_path, map_location=DEVICE))
            print(f"Optimizer state loaded from {optimizer_path}")
        else:
            print(f"Optimizer state checkpoint not found at {optimizer_path}")

        if os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                self.memory = pickle.load(f)
            print(f"Replay memory loaded from {memory_path}")
        else:
            print(f"Replay memory not found at {memory_path}")

        if os.path.exists(state_path):
            with open(state_path, 'rb') as f:
                training_state = pickle.load(f)
            self.n_games = training_state.get('n_games', 0)
            self.epsilon = training_state.get('epsilon', AgentConfig.EPSILON_START)
            self.record_score = training_state.get('record_score', 0)
            self.total_score = training_state.get('total_score', 0)
            self.plot_scores = training_state.get('plot_scores', [])
            self.plot_mean_scores = training_state.get('plot_mean_scores', [])
            print(f"Training state loaded from {state_path}")
            print(f"Resuming from Game {self.n_games} with Epsilon {self.epsilon:.4f}")
            return True
        else:
            print(f"Training state not found at {state_path}")
            return False
