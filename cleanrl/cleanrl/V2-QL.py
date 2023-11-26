import gymnasium as gym 
import numpy as np
import time
import math
# import cv2
import argparse
import os
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter

# render_mode = 'r'

# if render_mode == 'h':
#     env = gym.make("CartPole-v1", render_mode="human")
# elif render_mode == 'r':
#     env = gym.make("CartPole-v1", render_mode="rgb_array")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    

    args = parser.parse_args()
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk




# Learning Rate: learning rate is associated with how big you take a leap
lr = 0.1

#Discount Factor
gamma = 0.96

#Amount of iterations we are going to run until we see our model is trained
epochs = 100000
total_time = 0
total_reward = 0
prev_reward = 0
frames = []

Observation = [30, 30, 50, 50]
step_size = np.array([0.25, 0.25, 0.01, 0.1] )

# epsilon is associated with how random you take an action.
epsilon = 1

#exploration is decaying and we will get to a state of full exploitation
epsilon_decay_value = 0.99995


#randomly initializing values in our q table our q table


if __name__ == "__main__":

    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    
    args = parse_args()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    env = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    print(env.single_action_space.n)
    # action_spaces = env.action_space
    # total_actions = sum(np.prod(action_space.nvec) for action_space in action_spaces)
    # print(total_actions)

    # print("Total number of actions in the vectorized environment:", total_actions)
    q_table = np.random.uniform(low=0, high=1, size = (Observation + [env.single_action_space.n]))
    q_table.shape
    # print(q_table[0][0])

    def get_discrete_state(state):
        discrete_state = state[0]/step_size + np.array([15,10,1,10])
        discrete_state = np.clip(discrete_state, (0, 0, 0, 0), (Observation[0] - 1, Observation[1] - 1, Observation[2] - 1, Observation[3] - 1))
        return tuple(discrete_state.astype(np.int_))

    # ... (existing Q-learning code)

    writer = SummaryWriter(f"runs/{run_name}")
    #iterate through our epochs
    for epoch in range(epochs + 1): 
        #set the initial time, so we can calculate how much each action takes
        t_initial = time.time() 
        # print(env.observation_space.sample())
        #get the discrete state for the restarted environment, so we know what's going on
        discrete_state = get_discrete_state(env.reset()) 
        
        #we create a boolean that will tell us whether our game is running or not
        terminated = False  
        truncated = False
        
        #our reward is intialized at zero at the beginning of every eisode
        epoch_reward = 0 
        mean_reward = 0



        #Every 1000 epochs we have an episode
        if epoch % 1000  == 0: 
            print("Episode: " + str(epoch))

        while not (terminated or truncated): 
            #Now we are in our gameloop
            #if some random number is greater than epsilon, then we take the best possible action we have explored so far
            # print("env.single_action_space.n:   " +  str(env.single_action_space.n))
            # print("env.action_space:    " +  str(env.saction_space.n))
            # action = np.random.randint(0, env.single_action_space.n)
            # print("action:  " +  str(action))
            # print("env.action_space:" +  str(env.action_space))
            # print("env.action_space:" +  str(env.action_space))
            actions = []
            if isinstance(env.single_action_space, gym.spaces.Discrete):
                if np.random.random() > epsilon:
                    action = np.argmax(q_table[discrete_state])
                else:
                    action = np.random.randint(0, env.single_action_space.n)
                actions.append(action)
            elif isinstance(env.single_action_space, gym.spaces.MultiDiscrete):
                for i in range(env.single_action_space.n):
                    if np.random.random() > epsilon:
                        action = np.argmax(q_table[discrete_state])
                    else:
                        action = np.random.randint(0, env.single_action_space.nvec[i])
                    actions.append(action)

            #now we will intialize our new_state, reward, and done variables
            new_state, reward, terminated, truncated, _ = env.step(actions) 
        
            epoch_reward += reward 
            
            #we discretize our new state
            new_discrete_state = get_discrete_state(new_state)
            
            #we render our environment after 2000 steps
            # if epoch % 2000 == 0: 
            #     env.render()
            # if epoch % 10000 == 0:
            #     frame = env.render()
            #     frames.append(frame)


            
                # writer.add_scalar("charts/mean_time", mean, epoch)


            #if the game loop is still running update the q-table
            if not (terminated or truncated):
                max_new_q = np.max(q_table[new_discrete_state])

                current_q = q_table[discrete_state + (action,)]

                new_q = (1 - lr) * current_q + lr * (reward + gamma* max_new_q)

                q_table[discrete_state + (action,)] = new_q

            discrete_state = new_discrete_state
        # if our epsilon is greater than .05m , and if our reward is greater than the previous and if we reached past our 10000 epoch, we recalculate episilon
        
        if epsilon > 0.05: 
            if epoch_reward > prev_reward and epoch > 10000:
                epsilon = math.pow(epsilon_decay_value, epoch - 10000)

                if epoch % 500 == 0:
                    print("Epsilon: " + str(epsilon))

        #we calculate the final time
        tfinal = time.time() 
        
        #total epoch time
        episode_total = tfinal - t_initial 
        total_time += episode_total
        
        #calculate and update rewards
        total_reward += epoch_reward
        prev_reward = epoch_reward

        #every 1000 episodes print the average time and the average reward
        if epoch % 1000 == 0: 
            mean = total_time / 1000
            print("Time Average: " + str(mean))
            total_time = 0

            mean_reward = total_reward / 1000
            print("Mean Reward: " + str(mean_reward))
            total_reward = 0

        if epoch % 1000 == 0:
                writer.add_scalar("charts/episode_reward", epoch_reward, epoch)
                writer.add_scalar("charts/epsilon", epsilon, epoch)
                writer.add_scalar("charts/mean_reward", mean_reward, epoch)

env.close()

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('cartpole_episode.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))

# for frame in frames:
#     out.write(frame)

# out.release()