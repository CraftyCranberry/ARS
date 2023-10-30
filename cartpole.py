import gymnasium as gym 
import numpy as np
import time
import math
import cv2


env = gym.make("CartPole-v1", render_mode="rgb_array")
# env.reset()
# action = env.action_space.sample()
# observation, reward, terminated, truncated, info = env.step(action)
'''
print (env.step(action))
(array([ 0.04130387, -0.3475059 ,  0.03223267,  0.61519605], dtype=float32), 1.0, False, False, {})

ennnv.reset (array([ 0.02379748,  0.03448256, -0.04570031,  0.0096591 ], dtype=float32), {})

    ## Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    - 0: Push cart to the left
    - 1: Push cart to the right

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

'''


# Learning Rate: learning rate is associated with how big you take a leap
lr = 0.1

#Discount Factor
gamma = 0.96

#Amount of iterations we are going to run until we see our model is trained
epochs = 60000
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
q_table = np.random.uniform(low=0, high=1, size = (Observation + [env.action_space.n]))
q_table.shape
# print(q_table[0][0])

def get_discrete_state(state):
    discrete_state = state[0]/step_size + np.array([15,10,1,10])
    discrete_state = np.clip(discrete_state, (0, 0, 0, 0), (Observation[0] - 1, Observation[1] - 1, Observation[2] - 1, Observation[3] - 1))
    return tuple(discrete_state.astype(np.int_))


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

    #Every 1000 epochs we have an episode
    if epoch % 1000 == 0: 
        print("Episode: " + str(epoch))

    while not (terminated or truncated): 
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
        if epoch % 10000 == 0:
            frame = env.render()
            frames.append(frame)


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

env.close()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('cartpole_episode.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))

for frame in frames:
    out.write(frame)

out.release()