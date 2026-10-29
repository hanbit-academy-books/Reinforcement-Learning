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

def policy_evaluation_by_state_value(env,policy,gamma=0.99):
    V=np.zeros(env.observation_space.n)
    returns=[[] for i in range(env.observation_space.n)]
    for i in range(10000):
        epi=generate_episode(env,policy)
        R=0
        for t in range(len(epi)-2,-1,-1):
            s,r=epi[t][0],epi[t][2] # extracting state and reward
            R=gamma*R+r
            returns[s].append(R)
            V[s]=np.mean(returns[s])
    return V

env=gym.make('FrozenLake-v1',is_slippery=False,render_mode='ansi')

pi1=[[0.25,0.25,0.25,0.25] for i in range(env.observation_space.n)] # random policy
pi2=[[0,0.5,0.5,0] for i in range(env.observation_space.n)] # policy selecting 1(D) and 2(R) equally

np.set_printoptions(precision=5)
print('state value function of pi1:\n',policy_evaluation_by_state_value(env,pi1).reshape([4,4]))
print('state value function of pi2:\n',policy_evaluation_by_state_value(env,pi2).reshape([4,4]))
env.close()
