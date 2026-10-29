import gymnasium as gym
import numpy as np

n=100000 # number of episodes

pi2=np.array([[0,0.5,0.5,0],[0,0,1,0],[0,1,0,0],[1,0,0,0], # optimal policy
              [0,1,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0],
              [0,0,1,0],[0,0.5,0.5,0],[0,1,0,0],[0,0,0,0],
              [0,0,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,0]])

env=gym.make('FrozenLake-v1',render_mode='ansi',is_slippery=False)
env=gym.wrappers.TimeLimit(env,max_episode_steps=100)
episodes=[]
for _ in range(n):
    epi=[]
    state,info=env.reset()
    epi.append([None,state])
    while True:
        action=np.random.choice([0,1,2,3],1,p=pi2[state])[0] # action selection by optimal policy
        state1,reward,terminated,truncated,info=env.step(action)
        state=state1
        epi.append([action,reward,state])
        if terminated or truncated:
            break
    if epi[-1][1]!=1: print(epi); input('a)')
    episodes.append(epi)

env.close()

expected_return=sum([e[-1][1] for e in episodes])/n # calculation of expected return
print('expected return of optimal policy =',expected_return)
