import gymnasium as gym
import numpy as np

env=gym.make('FrozenLake-v1',render_mode='ansi',is_slippery=False)
print(env.observation_space) # state space
print(env.action_space)      # action space

state,info=env.reset()
print(env.render())
while True:
    action=np.random.choice([0,1,2,3],1,p=[0.25,0.25,0.25,0.25])[0] # random policy
    state1,reward,terminated,truncated,info=env.step(action)
    state=state1
    print(env.render())
    if terminated or truncated:
        break

env.close()
