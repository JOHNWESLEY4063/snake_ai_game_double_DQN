# terminal/train.py
import pygame
import torch
import numpy as np
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os

# Import modules from the same terminal folder, adjusted for snake_game_ai.py
from snake_game_ai import SnakeGameAI, Direction, Point
from agent import Agent
from model import Linear_QNet, QTrainer
from plots import set_seeds, plot # Import plot function
from config import AgentConfig, GameConfig, NetworkConfig, LOG_DIR, MODEL_DIR # Import all from config

def train_terminal_ddqn():
    parser = argparse.ArgumentParser(description="Train a Double DQN agent for Snake Game (Terminal Version).")
    parser.add_argument("--obstacles", action="store_true",
                        help="Enable obstacles in the game environment.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the last saved checkpoint.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    set_seeds(args.seed)

    # Initialize TensorBoard writer
    run_name = f"{'obstacles' if args.obstacles else 'normal'}_ddqn_terminal_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(os.path.join(LOG_DIR, run_name)) # LOG_DIR from config.py in this folder
    print(f"TensorBoard logs visible at: {os.path.join(LOG_DIR, run_name)}")

    game = SnakeGameAI(enable_obstacles=args.obstacles,
                       w=GameConfig.WIDTH,
                       h=GameConfig.HEIGHT)

    agent = Agent()

    if args.resume:
        print("Attempting to resume training...")
        if agent.load_checkpoint(file_prefix="final_training_state"):
            print("Training resumed successfully from final_training_state.")
        elif agent.load_checkpoint(file_prefix=f"checkpoint_game_{agent.n_games}"):
             print(f"Training resumed successfully from game {agent.n_games} checkpoint.")
        else:
            print("No suitable checkpoint found. Starting new training session.")
            agent = Agent()
            agent.target_model.load_state_dict(agent.model.state_dict())
            agent.target_model.eval()
    else:
        print("Starting new training session.")

    global_step = 0

    while True:
        state_old = game.get_state()
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = game.get_state()

        loss_short = agent.train_short_memory(state_old, final_move, reward, state_new, done)
        writer.add_scalar('Loss/Short_Memory_Loss', loss_short, global_step)

        agent.remember(state_old, final_move, reward, state_new, done)

        global_step += 1

        if done:
            game.reset()
            agent.n_games += 1

            if len(agent.memory) > AgentConfig.BATCH_SIZE * 2:
                loss_long = agent.train_long_memory()
                writer.add_scalar('Loss/Long_Memory_Loss', loss_long, agent.n_games)
            else:
                loss_long = 0

            if agent.n_games % AgentConfig.UPDATE_TARGET_EVERY_GAMES == 0:
                agent.update_target_network()

            agent.plot_scores.append(score)
            agent.total_score += score
            mean_score = agent.total_score / agent.n_games
            agent.plot_mean_scores.append(mean_score)

            if score > agent.record_score:
                agent.record_score = score
                agent.model.save(file_name=AgentConfig.MAIN_MODEL_FILE, model_dir=MODEL_DIR) # MODEL_DIR from config.py in this folder
                agent.target_model.save(file_name=AgentConfig.TARGET_MODEL_FILE, model_dir=MODEL_DIR) # MODEL_DIR from config.py in this folder
                print(f"New record score! Model saved. Score: {score}")

            current_epsilon = max(AgentConfig.EPSILON_MIN, AgentConfig.EPSILON_START * (AgentConfig.EPSILON_DECAY_RATE ** agent.n_games))
            
            print(f'Game {agent.n_games:5} | Score: {score:3} | Record: {agent.record_score:3} | Mean Score: {mean_score:.2f} | Epsilon: {current_epsilon:.4f} | Mem: {len(agent.memory)}')

            writer.add_scalar('Score/Episode_Score', score, agent.n_games)
            writer.add_scalar('Score/Mean_Score', mean_score, agent.n_games)
            writer.add_scalar('Hyperparameters/Epsilon', current_epsilon, agent.n_games)
            writer.add_scalar('Game_Stats/Episode_Length', game.frame_iteration, agent.n_games)
            writer.add_scalar('Game_Stats/Record_Score', agent.record_score, agent.n_games)
            writer.add_scalar('Game_Stats/Memory_Size', len(agent.memory), agent.n_games)

            # Call Matplotlib plot after each episode
            plot(agent.plot_scores, agent.plot_mean_scores)

            if agent.n_games % AgentConfig.SAVE_CHECKPOINT_EVERY_GAMES == 0:
                agent.save_checkpoint(file_prefix=f"checkpoint_game_{agent.n_games}")
                print(f"Checkpoint saved at game {agent.n_games}")

    writer.close()
    print("Training complete.")
    agent.save_checkpoint(file_prefix="final_training_state")

if __name__ == '__main__':
    train_terminal_ddqn()
