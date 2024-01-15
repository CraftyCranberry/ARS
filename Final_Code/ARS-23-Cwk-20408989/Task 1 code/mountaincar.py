import gymnasium as gym
import numpy as np
import time
import math
import cv2
render_mode = 'h'

if render_mode == 'h':
    env = gym.make("MountainCar-v0", render_mode="human")
elif render_mode == 'r':
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
# env = gym.make("CartPole-v1", render_mode="rgb_array")
# print(env.action_space.n)
# print(env.observation_space.sample())


'''
## Observation Space

    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:

    | Num | Observation                          | Min   | Max  | Unit         |
    |-----|--------------------------------------|-------|------|--------------|
    | 0   | position of the car along the x-axis | -1.2  | 0.6  | position (m) |
    | 1   | velocity of the car                  | -0.07 | 0.07 | velocity (v) |

    ## Action Space

    There are 3 discrete deterministic actions:

    - 0: Accelerate to the left
    - 1: Don't accelerate
    - 2: Accelerate to the right
'''

# Learning Rate: learning rate is associated with how big you take a leap
lr = 0.01

#Discount Factor
gamma = 0.99

#Amount of iterations we are going to run until we see our model is trained
epochs = 1
total_time = 0
total_reward = 0
prev_reward = 0
frames = []

Observation = [30, 30]
step_size = np.array([0.01, 0.01])

# epsilon is asdsociated with how random you take an action.
epsilon = 1

#exploration is decaying and we will get to a state of full exploitation
epsilon_decay_value = 0.999999

#randomly initializing values in our q table our q table
q_table = np.random.uniform(low=0, high=1, size = (Observation + [env.action_space.n]))
q_table.shape



def get_discrete_state(state):
    discrete_state = state[0]/step_size + np.array([15,10])
    discrete_state = np.clip(discrete_state, (0, 0), (Observation[0] - 1, Observation[1] - 1))
    return tuple(discrete_state.astype(np.int_))




for epoch in range(epochs + 1):
    #set the initial time, so we can calculate how much each action takes
    t_initial = time.time()
    # print(env.observation_space.sample())
    #get the discrete state for the restarted environment, so we know what's going on
    discrete_state = get_discrete_state(env.reset())
    # print (discrete_state)
    #we create a boolean that will tell us whether our game is running or not
    terminated = False
    truncated = False
    step_counter = 0  # Add a step counter

    #our reward is intialized at zero at the beginning of every eisode
    epoch_reward = 0

    #Every 1000 epochs we have an episode
    if epoch % 1 == 0:
        print("Episode: " + str(epoch))
    
    # print("Epoch Reward: {}".format(epoch_reward))
    
    while not (terminated):
        #Now we are in our gameloop
        #if some random number is greater than epsilon, then we take the best possible action we have explored so far
        if np.random.random() > epsilon:

            action = np.argmax(q_table[discrete_state])

        #else, we will explore and take a random action
        else:

            action = np.random.randint(0, env.action_space.n)

        #now we will intialize our new_state, reward, and done variables
        new_state, reward, terminated, truncated, _ = env.step(action)


        epoch_reward += reward
        #we discretize our new state
        new_discrete_state = get_discrete_state(new_state)

        #we render our environment after 2000 steps
        # if epoch % 2000 == 0:
        #     env.render()


        

        #if the game loop is still running update the q-table
        if not (terminated):
            max_new_q = np.max(q_table[new_discrete_state])

            current_q = q_table[discrete_state + (action,)]

            new_q = (1 - lr) * current_q + lr * (reward + gamma* max_new_q)

            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state
    # if our epsilon is greater than .05m , and if our reward is greater than the previous and if we reached past our 10000 epoch, we recalculate episilon

    if epsilon > 0.05:
        if epoch_reward > prev_reward and epoch > 10000:
            epsilon = math.pow(epsilon_decay_value, epoch - 10000)

            if epoch % 1 == 0:
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
        mean = total_time / 1
        print("Time Average: " + str(mean))
        total_time = 0

        mean_reward = total_reward / 1
        print("Mean Reward: " + str(mean_reward))
        print()
        total_reward = 0

    step_counter += 1
    if step_counter >= 3000:
      terminated = True
    if epoch % 1 == 0:
            frame = env.render()
            frames.append(frame)

env.close()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('MountainCar_episode.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))

for frame in frames:
    out.write(frame)

out.release()
