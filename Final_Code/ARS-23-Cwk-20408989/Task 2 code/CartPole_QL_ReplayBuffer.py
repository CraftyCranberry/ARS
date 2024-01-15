#This code was adapted from Johnnyode8's GitHub repo: https://github.com/johnnycode8/gym_solutions

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import argparse
import random
import os
from distutils.util import strtobool
import datetime
from torch.utils.tensorboard import SummaryWriter
import shutil
from google.colab import drive

# Mount Google Drive for persistent storage
drive.mount('/content/drive')


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = []
        self.size = size
        self.next_idx = 0

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)

        if self.next_idx >= len(self.buffer):
            self.buffer.append(data)
        else:
            self.buffer[self.next_idx] = data

        self.next_idx = (self.next_idx + 1) % self.size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)



def run(is_training=True, render=False):
    # Hyperparameters
    learning_rate_a = 0.25  # Alpha: learning rate
    discount_factor_g = 0.99  # Gamma: discount factor
    epsilon = 1  # Exploration rate: 100% random actions initially
    epsilon_decay_rate = 0.0001  # Rate at which exploration decreases
    batch_size = 10  # Define your batch size
    rng = np.random.default_rng()  # Random number generator

    episode = 0
    rewards_per_episode = []

    # Logging and saving setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"CartPole_QL_ReplayBuffer_LR_{learning_rate_a}_GAMMA_{discount_factor_g}_EPS_{epsilon}_DECAY_{epsilon_decay_rate}_{timestamp}_BatchSize=1"
    local_log_path = f"final_code/runs/{run_name}"
    writer = SummaryWriter(local_log_path)
    source_folder = f"final_code/runs/{run_name}"
    destination_folder = '/content/drive/My Drive/RL/final_code/runs/'

    # Environment setup
    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    # Replay buffer initialization
    buffer_size = 10000
    replay_buffer = ReplayBuffer(buffer_size)

    # State space discretization
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    # Q-table initialization
    if is_training:
        q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n))
    else:
        # Load Q-table if not training
        file_path = '/content/drive/My Drive/RL/final_code/runs/q_tables/cartpole_QL_ReplayBuffer.pkl'
        with open(file_path, 'rb') as f:
            q = pickle.load(f)

    

    while(True):

        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False
        total_reward = 0
        episode_length = 0

        while not terminated and total_reward < 10000:
            episode_length += 1

            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])
            
            # Take action and observe new state and reward
            new_state,reward,terminated,_,_ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av= np.digitize(new_state[3], ang_vel_space)

            replay_buffer.add(state, action, reward, new_state, terminated)

           
            if is_training and len(replay_buffer) >= batch_size:
                mini_batch = replay_buffer.sample(batch_size)

                for state, action, reward, next_state, done in mini_batch:
                    # Manually digitize each component of the state and next_state
                    state_p = np.digitize(state[0], pos_space)
                    state_v = np.digitize(state[1], vel_space)
                    state_a = np.digitize(state[2], ang_space)
                    state_av = np.digitize(state[3], ang_vel_space)

                    next_state_p = np.digitize(next_state[0], pos_space)
                    next_state_v = np.digitize(next_state[1], vel_space)
                    next_state_a = np.digitize(next_state[2], ang_space)
                    next_state_av = np.digitize(next_state[3], ang_vel_space)

                    # Calculate the target value
                    if done:
                        target = reward
                    else:
                        target = reward + discount_factor_g * np.max(q[next_state_p, next_state_v, next_state_a, next_state_av, :])

                    # Update Q-value
                    q[state_p, state_v, state_a, state_av, action] = (1 - learning_rate_a) * q[state_p, state_v, state_a, state_av, action] + learning_rate_a * target
            
            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av= new_state_av

            total_reward+=reward


            if not is_training and total_reward%100==0:
                print(f'Episode: {episode}  Rewards: {total_reward}')

        # Tensorboard logging
        if is_training and episode % 10 == 0:
            writer.add_scalar("charts/episodic_return", total_reward, episode)
            writer.add_scalar("charts/epsilon", epsilon, episode)
            writer.add_scalar("charts/episodic_length", episode_length, episode)
            writer.add_scalar("charts/learning_rate", learning_rate_a, episode)
            writer.add_scalar("charts/gamma", discount_factor_g, episode)
            writer.add_scalar("charts/epsilon_decay_rate", epsilon_decay_rate, episode)

        epsilon = max(epsilon - epsilon_decay_rate, 0)

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
    plt.savefig('cartpole_QL_ReplayBuffer.png')

if __name__ == '__main__':

    run(is_training=True, render=False)

