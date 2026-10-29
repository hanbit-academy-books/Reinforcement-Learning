import gymnasium as gym
import numpy as np

gamma=0.9 # discount rate

def policy_evaluation(env,policy):
    V=np.zeros(env.observation_space.n) # table to store the value function
    while True:
        oldV=V.copy()
        for state in range(env.observation_space.n):
            v=0
            for action,action_prob in enumerate(policy[state]): # Bellman equation
                prob,next_state,reward,terminated=env.unwrapped.P[state][action][0]
                v=v+action_prob*(reward+gamma*V[next_state])
            V[state]=v
        if max(np.abs(V-oldV))<1e-8:
            break
    return V

env=gym.make('FrozenLake-v1',is_slippery=False,render_mode='ansi')

pi1=np.ones((env.observation_space.n,env.action_space.n))/env.action_space.n # random policy
V=policy_evaluation(env,pi1)
print('Value function of pi1 (random policy):\n',np.round(V.reshape([4,4]),4))
