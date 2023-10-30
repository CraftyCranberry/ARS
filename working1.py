import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import sys
import subprocess
env = gym.make("CartPole-v1",render_mode="rgb_array")
env.reset()

'''

Description of state formations, we divide the the state values into discrete bins. Like :
Cart x-position (-2.4 : 2.4) is divided into 48 bins of length size 0.1
Similarly pole angle (-12, 12) (in degrees) is divided into 24 bins of size 1
Cart velocity (-2.5, 2.5) is divided into 10 bins of 0.5 size
Pole angular velocity (-12, 12) (in degrees/s) is divided into 12 bins of size 2

'''


def show(t):
    img = env.render()
    plt.imshow(img)
    plt.savefig(f"./{t}.png")

q = np.zeros((12*8*5*6,2))
e = np.zeros((12*8*5*6,2))
terminated = False
for _ in range(20000):
    sys.stdout.write(f"\r{_+1} iteration")
    env.reset()
    terminated = 0
    state = _%(12*8*5*6)
    while not terminated:
        if q[state][0]==0 and q[state][1]==0:
            action = np.random.choice(np.array([0,1]))
        else:
            action = np.argmax(q[state, :]) if np.random.random() > 0.2/(1.1**(_//1000)) else np.random.choice(np.array([0,1]))
        e *= 0.8
        e[state][action] += 1
        observation, reward, terminated, truncated, info = env.step(action)
        # Q updates
        # Calculating new state
        pos, vel, apos, avel = observation
        vel = max(-2.5, vel)
        vel = min(2.5, vel)
        avel = max(-12, avel)
        avel = min(12, avel)
        next_state = 8*5*6*min(11, int(5*pos/2+6)) + 5*6*min(int((apos*180/np.pi+12)/3),7) + 6*min(int(vel+2.5),4) + min(int((avel*180/np.pi)//4+3), 5)
        
        q = q + 0.2*(reward+np.argmax(q[next_state])-q)*e
    
    q = q + 0.9*(-1-q)*(e/np.sum(e))
np.save("./q.npy", q)
env.reset()
terminated = 0
t = 0
images = []
state = np.random.choice(12*8*5*6)
while not terminated and t<1000:
    t += 1
    show(t)
    images.append(imageio.imread(f"./{t}.png"))
    subprocess.run(['rm', str(t)+".png"])
    action = np.argmax(q[state, :])
    observation, reward, terminated, truncated, info = env.step(action)
    pos, vel, apos, avel = observation
    vel = max(-2.5, vel)
    vel = min(2.5, vel)
    avel = max(-12, avel)
    avel = min(12, avel)
    state = 8*5*6*min(11, int(5*pos/2+6)) + 5*6*min(int((apos*180/np.pi+12)/3),7) + 6*min(int(vel+2.5),4) + min(int((avel*180/np.pi)//4+3), 5)
print(np.sum(q.T[0]), np.sum(q.T[1]))
print(t)
output_path = 'output.gif'
imageio.mimsave(output_path, images, duration=0.5)
env.close()