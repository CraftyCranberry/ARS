# Solutions for Gymnasium Reinforcement Learning Library
## Gymnasium library
The [Gymnasium library](https://gymnasium.farama.org/) is supported on Linux and Mac OS. If you have trouble installing on Windows (Failed building wheels for box2d-py?), check out my guide [Install Gymnasium on Windows](https://youtu.be/gMgj4pSHLww)

## Frozen Lake 8x8
**frozen_lake_q.py**  
Solves the [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) 8x8 map with Q-Learning.  
Companion Youtube video: [Solve FrozenLake-v1 8x8 Tutorial](https://youtu.be/ZhoIgo3qqLU)

## Frozen Lake 8x8 Enhanced
**frozen_lake_qe.py**  
This file is almost identical to frozen_lake_q.py above, except this uses the frozen_lake_enhanced.py environment.  

**frozen_lake_enhanced.py**  
This is the FrozenLake-v1 environment "enhanced" to help you better understand Q-Learning. Features:
* The Q values are overlayed on top of each cell of the map. Visually see the Q values update in realtime!
* The map is enlarged to fill the whole screen so that it is easier to read the Q values.
* Shortcut keys to speed up or slow down the animation.

Companion Youtube video: [See Q-Learning in Realtime on FrozenLake-v1](https://youtu.be/1W_LOB-0IEY)

## Mountain Car
**mountain_car_q.py**  
Solves the [MountainCar-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car/) environment with Q-Learning.  
Companion Youtube video: [Solve MountainCar-v0 Tutorial](https://youtu.be/_SWnNhM5w-g)

## Cart Pole
**cartpole_q.py**  
Solves the [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment with Q-Learning.  
Companion Youtube video: [Solve CartPole-v1 Tutorial](https://youtu.be/2u1REHeHMrg)

## Humanoid
**sb3.py**  
Solves the [Humanoid-v4](https://gymnasium.farama.org/environments/mujoco/humanoid/) environment with Stable Baselines3 Soft Actor-Critic.  
Companion Youtube video: [Solve Humanoid-v4 Tutorial](https://youtu.be/OqvXHi_QtT0)