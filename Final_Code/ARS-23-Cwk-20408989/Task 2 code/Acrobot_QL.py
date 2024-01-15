#This code was adapted from Johnnyode8's GitHub repo: https://github.com/johnnycode8/gym_solutions

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

import time
import argparse
import os
from distutils.util import strtobool
import datetime
from torch.utils.tensorboard import SummaryWriter
import shutil
from google.colab import drive

# Mount Google Drive for storage
drive.mount('/content/drive')

def run(is_training=True, render=False):
    # Hyperparameters
    learning_rate_a = 0.25  # Alpha: learning rate
    discount_factor_g = 0.99  # Gamma: discount factor
    epsilon = 1  # Exploration rate: 100% random actions initially
    epsilon_decay_rate = 0.0001  # Rate at which exploration decreases
    rng = np.random.default_rng()  # Random number generator

    rewards_per_episode = []
    max_rewards = 0
    mean_rewards = 0
    episode = 0

    # Logging and saving setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Acrobot_QL_LR_{learning_rate_a}_GAMMA_{discount_factor_g}_EPS_{epsilon}_DECAY_{epsilon_decay_rate}_{timestamp}"
    local_log_path = f"final_code/runs/{run_name}"
    writer = SummaryWriter(local_log_path)
    source_folder = f"final_code/runs/{run_name}"
    destination_folder = '/content/drive/My Drive/RL/final_code/runs/'

    # Environment setup
    env = gym.make('Acrobot-v1', render_mode='human' if render else None)

    # Discretizing the state space
    cos_t1 = np.linspace(-1, 1, 10)
    sin_t1 = np.linspace(-1, 1, 10)
    cos_t2 = np.linspace(-1, 1, 10)
    sin_t2 = np.linspace(-1, 1, 10)
    ang_vel_t1 = np.linspace(-12.567, 12.567, 20)
    ang_vel_t2 = np.linspace(-28.274, 28.274, 40)

    # Q-table initialization
    if(is_training):
        q = np.zeros((len(cos_t1)+1, len(sin_t1)+1, len(cos_t2)+1, len(sin_t2)+1, len(ang_vel_t1)+1, len(ang_vel_t2)+1, env.action_space.n))
    else:
        # Load Q-table if not training
        file_path = '/content/drive/My Drive/RL/final_code/runs/q_tables/acrobot.pkl'
        with open(file_path, 'rb') as f:
            q = pickle.load(f)
   
    # Main loop
    while(True):
        # Reset and discretize initial state
        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_cos_t1 = np.digitize(state[0], cos_t1)
        state_sin_t1 = np.digitize(state[1], sin_t1)
        state_cos_t2 = np.digitize(state[2], cos_t2)
        state_sin_t2 = np.digitize(state[3], sin_t2)
        state_av_t1 = np.digitize(state[4], ang_vel_t1)
        state_av_t2 = np.digitize(state[5], ang_vel_t2)

        terminated = False
        total_reward = 0
        episode_length = 0

        while(not terminated or episode_length < -200):
            episode_length += 1

            if is_training and rng.random() < epsilon:
                # Choose random action  (0=apply -1 torque, 1=apply 0 torque, 2=apply 1 torque)
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_cos_t1, state_sin_t1, state_cos_t2, state_sin_t2, state_av_t1, state_av_t2, :])
            
            # Take action and observe new state and reward
            new_state,reward,terminated,_,_ = env.step(action)
            new_state_cos_t1 = np.digitize(new_state[0], cos_t1)
            new_state_sin_t1 = np.digitize(new_state[1], sin_t1)
            new_state_cos_t2 = np.digitize(new_state[2], cos_t2)
            new_state_sin_t2 = np.digitize(new_state[3], sin_t2)
            new_state_av_t1 = np.digitize(new_state[4], ang_vel_t1)
            new_state_av_t2 = np.digitize(new_state[5], ang_vel_t2)

            # Q-value update
            if is_training:
                q[state_cos_t1, state_sin_t1, state_cos_t2, state_sin_t2, state_av_t1, state_av_t2, action] = q[state_cos_t1, state_sin_t1, state_cos_t2, state_sin_t2, state_av_t1, state_av_t2, action] + learning_rate_a * (
                    reward + discount_factor_g*np.max(q[new_state_cos_t1, new_state_sin_t1, new_state_cos_t2, new_state_sin_t2, new_state_av_t1, new_state_av_t2,:]) - q[state_cos_t1, state_sin_t1, state_cos_t2, state_sin_t2, state_av_t1, state_av_t2, action])
            
            state = new_state
            state_cos_t1 = new_state_cos_t1
            state_sin_t1 = new_state_sin_t1
            state_cos_t2 = new_state_cos_t2
            state_sin_t2 = new_state_sin_t2
            state_av_t1 = new_state_av_t1
            state_av_t2 = new_state_av_t2

            total_reward += reward

             # Monitoring training progress
            if not is_training and total_reward % 100 == 0:
                print(f'Episode: {episode}  Rewards: {total_reward}')

        # Tensorboard logging
        if is_training and episode % 10 == 0:
            writer.add_scalar("charts/episodic_return", total_reward, episode)
            writer.add_scalar("charts/epsilon", epsilon, episode)
            writer.add_scalar("charts/episodic_length", episode_length, episode)
            writer.add_scalar("charts/learning_rate", learning_rate_a, episode)
            writer.add_scalar("charts/gamma", discount_factor_g, episode)
            writer.add_scalar("charts/epsilon_decay_rate", epsilon_decay_rate, episode)

        epsilon = max(epsilon - epsilon_decay_rate, 0)   # Epsilon decay

        # Save Q-table and logs
        if is_training and episode % 100 == 0:
            print("Saving Q-Table")
            file_path = f"/content/drive/My Drive/RL/final_code/runs/q_tables/{run_name}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(q, f)
            try:
                destination_path = f"{destination_folder}/{source_folder.split('/')[-1]}"
                shutil.copytree(source_folder, destination_path, dirs_exist_ok=True)
            except Exception as e:
                print(f"Error in copying files: {e}")

        rewards_per_episode.append(total_reward)

        # if len(rewards_per_episode) >= 100:
        #     mean_rewards = np.mean(rewards_per_episode[-100:])
        #     # Break if mean reward over the last 200 episodes is greater than 500
        #     if mean_rewards > 100:
        #         break

         #Break after 15000 episodes
        if episode > 15000: 
            break

        if is_training and episode % 100 == 0:
            mean_rewards = np.mean(rewards_per_episode[-1000:])
            print(f'Episode: {episode} {total_reward}  Epsilon: {epsilon:.2f}  Mean Rewards: {mean_rewards:.1f}')

        episode += 1

    env.close()

    # Plotting rewards
    mean_rewards = [np.mean(rewards_per_episode[max(0, t-100):(t+1)]) for t in range(episode)]
    plt.plot(mean_rewards)
    plt.savefig(f'acrobot_QL.png')

if __name__ == '__main__':

    run(is_training=True, render=False)

