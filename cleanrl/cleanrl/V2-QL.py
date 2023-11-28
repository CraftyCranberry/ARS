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
    parser.add_argument("--num-episodes", type=int, default="10000",
        help="the number of episodes")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--epsilon", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--epsilon-decay-rate", type=float, default=0.99995,
        help="the ending epsilon for exploration")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")

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





lr = 0
gamma = 0
epsilon = 0
epsilon_decay_value = 0

#Amount of iterations we are going to run until we see our model is trained
epochs = 0
total_time = 0
total_reward = 0
prev_reward = 0
frames = []
actions = []   

Observation = [30, 30, 50, 50]
# step_size = np.array([0.25, 0.25, 0.01, 0.1] )
step_size = 0.1

# epsilon is associated with how random you take an action.


#exploration is decaying and we will get to a state of full exploitation



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
    epochs = args.num_episodes
    lr = args.learning_rate
    gamma = args.gamma
    epsilon = args.epsilon
    epsilon_decay_value = args.epsilon_decay_rate


    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    
    env = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    # print(env.single_action_space.n)
    # action_spaces = env.action_space
    # total_actions = sum(np.prod(action_space.nvec) for action_space in action_spaces)
    # print(total_actions)

    # print("Total number of actions in the vectorized environment:", total_actions)
    q_table = np.random.uniform(low=0, high=1, size = (Observation + [env.single_action_space.n]))
    q_table.shape
    # print(q_table[0][0])

    def get_discrete_state(ent, step_size):
        observation_space = env.observation_space
        state = ent[0]
        discrete_state = state[0] / step_size + (observation_space.high - observation_space.low) / 2
        discrete_state = np.clip(discrete_state, observation_space.low, observation_space.high - 1)
        return tuple(discrete_state.astype(np.int_))
    # ... (existing Q-learning code)

    writer = SummaryWriter(f"runs/{run_name}")
    #iterate through our epochs
    for epoch in range(epochs + 1): 
        # print(env.reset)
        #set the initial time, so we can calculate how much each action takes
        t_initial = time.time() 
        # print(env.observation_space.sample())
        #get the discrete state for the restarted environment, so we know what's going on
        # print(env.reset())
        discrete_state = get_discrete_state(env.reset(), step_size) 
        
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
            #if some random number is greater than epsilon, then we take the best possible action we have explored so far            actions = []
            if isinstance(env.single_action_space, gym.spaces.Discrete):
                if np.random.random() > epsilon:
                    # print("bat")
                    # print(f"Discrete State: {discrete_state}")
                    action = np.argmax(q_table[discrete_state])
                else:
                    # print("cat  ")
                    action = np.random.randint(0, env.single_action_space.n)
                # print(f"Action: {action}")
                actions.append(action)
                # print(f"Action: {action}")
            elif isinstance(env.single_action_space, gym.spaces.MultiDiscrete):
                for i in range(env.single_action_space.n):
                    if np.random.random() > epsilon:
                        action = np.argmax(q_table[discrete_state])
                    else:
                        action = np.random.randint(0, env.single_action_space.nvec[i])
                    # print(f"Action: {action}")
                    actions.append(action)
                    # print(f"Action: {action}")

            # if action > 1:
            #     action = 1

            # print(f"Action selected: {action}")
            #now we will intialize our new_state, reward, and done variables
            new_state, reward, terminated, truncated, _ = env.step(actions) 
        
            epoch_reward += reward 
            
            #we discretize our new state
            # new_discrete_state = get_discrete_state(new_state)
            new_discrete_state = get_discrete_state(new_state, step_size) 
            # print(new_discrete_state)
            # print(new_discrete_state)
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
        # print ('Epsilonss: ' + str (epsilon))
        if epsilon > 0.05: 
            if epoch_reward > prev_reward and epoch > 1000:
                # print ('Epsilon: ' + str (epsilon))
                epsilon = math.pow(epsilon_decay_value, epoch - 1000)

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
        if epoch % 1 == 0: 
            mean = total_time / 1000
            print("Time Average: " + str(mean))
            total_time = 0

            mean_reward = total_reward / 1000
            writer.add_scalar("charts/mean_reward", mean_reward, epoch)
            print("Mean Reward: " + str(mean_reward))
            total_reward = 0

        writer.add_scalar("charts/episodic_return", epoch_reward, epoch)
        writer.add_scalar("charts/epsilon", epsilon, epoch)
        writer.add_scalar("charts/episodic_length", episode_total, epoch)
        writer.add_scalar("charts/learning_rate", lr, epoch)
        writer.add_scalar("charts/epsilon", epsilon, epoch)
        writer.add_scalar("charts/gamma", gamma, epoch)
                

env.close()

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('cartpole_episode.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))

# for frame in frames:
#     out.write(frame)

# out.release()