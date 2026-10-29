import gymnasium as gym
import numpy as np

gamma=0.9 # discount rate

def policy_iteration(env):
    V=np.zeros(env.observation_space.n)
    pi=[0 for i in range(env.observation_space.n)] # initialize policy to action 0
    while True:
        while True: # E (policy evaluation)
            oldV=V.copy()
            for state in range(env.observation_space.n):
                action=pi[state]
                prob,next_state,reward,terminated=env.unwrapped.P[state][action][0]
                v=reward+gamma*V[next_state]
                V[state]=v
            if max(np.abs(V-oldV))<1e-8:
                break

        converged=True # I (policy improvement)
        for state in range(env.observation_space.n):
            old_action=pi[state]
            q=np.zeros(env.action_space.n)
            for action in range(env.action_space.n):
                prob,next_state,reward,terminated=env.unwrapped.P[state][action][0]
                q[action]=reward+gamma*V[next_state]
            pi[state]=np.argmax(q)
            if pi[state]!=old_action:
                converged=False
        if converged:
            break
    return V,pi

env=gym.make('FrozenLake-v1',is_slippery=False,render_mode='ansi')

V,pi=policy_iteration(env)
print('optimal policy:\n',np.array(pi).reshape([4,4]))
print('optimal value function:\n',np.round(V.reshape([4,4]),4))
