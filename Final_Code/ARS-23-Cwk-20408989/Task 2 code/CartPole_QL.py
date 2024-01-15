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

# Mount Google Drive for persistent storage
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
    run_name = f"CartPole_QL_LR_{learning_rate_a}_GAMMA_{discount_factor_g}_EPS_{epsilon}_DECAY_{epsilon_decay_rate}_{timestamp}"
    local_log_path = f"final_code/runs/{run_name}"
    writer = SummaryWriter(local_log_path)
    source_folder = f"final_code/runs/{run_name}"
    destination_folder = '/content/drive/My Drive/RL/final_code/runs/'

    # Environment setup
    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    # Discretizing the state space
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

     # Q-table initialization
    if is_training:
        q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n))
    else:
        # Load Q-table if not training
        file_path = '/content/drive/My Drive/RL/final_code/runs/q_tables/cartpole_QL.pkl'
        with open(file_path, 'rb') as f:
            q = pickle.load(f)

    # Main loop
    while(True):
         # Reset and discretize initial state
        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False
        total_reward = 0
        episode_length = 0

        while(not terminated and total_reward < 10000):
            episode_length += 1

            if is_training and rng.random() < epsilon:
                # Choose random action  (0=go left, 1=go right)
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])
            
            # Take action and observe new state and reward
            new_state,reward,terminated,_,_ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av= np.digitize(new_state[3], ang_vel_space)

            # Q-value update
            if is_training:
                q[state_p, state_v, state_a, state_av, action] = q[state_p, state_v, state_a, state_av, action] + learning_rate_a * (
                    reward + discount_factor_g*np.max(q[new_state_p, new_state_v, new_state_a, new_state_av,:]) - q[state_p, state_v, state_a, state_av, action])

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av= new_state_av

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
        
        epsilon = max(epsilon - epsilon_decay_rate, 0)  # Epsilon decay

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
    plt.savefig('cartpole_QL.png')

if __name__ == '__main__':

    run(is_training=True, render=False)
