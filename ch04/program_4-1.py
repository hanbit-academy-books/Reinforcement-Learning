import gymnasium as gym
import numpy as np

def generate_episode(env,policy):
    epi=[]
    s,info=env.reset()
    while True:
        a=np.random.choice([0,1,2,3],1,p=policy[s])[0]
        s1,r,terminated,truncated,info=env.step(a)
        epi.append([s,a,r]) # appending [state,action,reward]
        s=s1
        if terminated or truncated:
            epi.append([s1,None,None])
            break
    return epi

env=gym.make('FrozenLake-v1',is_slippery=False,render_mode='ansi')

pi1=np.ones((env.observation_space.n,env.action_space.n))/env.action_space.n

episode=generate_episode(env,pi1)
print('episode generated with random policy:\n',np.array(episode))
env.close()
