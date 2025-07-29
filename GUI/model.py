# ddqn_gui/model.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.optim.lr_scheduler import StepLR
# Import MODEL_DIR directly from config in this folder
from config import NetworkConfig, AgentConfig, DEVICE, MODEL_DIR

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
        self.to(DEVICE)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth', model_dir=MODEL_DIR): # Use the directly imported MODEL_DIR
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        file_path = os.path.join(model_dir, file_name)
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done, target_model=None):
        state = torch.tensor(state, dtype=torch.float).to(DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float).to(DEVICE)
        action = torch.tensor(action, dtype=torch.long).to(DEVICE)
        reward = torch.tensor(reward, dtype=torch.float).to(DEVICE)
        done = torch.tensor(done, dtype=torch.bool).to(DEVICE)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                if target_model:
                    with torch.no_grad():
                        next_state_q_values = self.model(next_state[idx].unsqueeze(0))
                        best_action_idx = torch.argmax(next_state_q_values).item()
                        Q_next = target_model(next_state[idx].unsqueeze(0))[0][best_action_idx]
                    Q_new = reward[idx] + self.gamma * Q_next
                else:
                    with torch.no_grad():
                        Q_next = self.model(next_state[idx].unsqueeze(0))
                    Q_new = reward[idx] + self.gamma * torch.max(Q_next)
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()
