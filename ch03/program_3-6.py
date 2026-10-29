import gymnasium as gym
import numpy as np

gamma=0.9 # discount rate

def value_iteration(env):
    V=np.zeros(env.observation_space.n)
    while True: # estimate optimal value function
        oldV=V.copy()
        for state in range(env.observation_space.n):
            q=np.zeros(env.action_space.n)
            for action in range(env.action_space.n):
                for prob,next_state,reward,terminated in env.unwrapped.P[state][action]:
                    q[action]=q[action]+prob*(reward+gamma*V[next_state])
            V[state]=np.max(q)
        if max(np.abs(V-oldV))<1e-8:
            break

    pi=env.observation_space.n*[None] # derive optimal policy from optimal value function
    for state in range(env.observation_space.n):
        q=np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for prob,next_state,reward,terminated in env.unwrapped.P[state][action]:
                q[action]=q[action]+prob*(reward+gamma*V[next_state])
        pi[state]=np.argmax(q)
    return pi,V

env=gym.make('FrozenLake-v1',is_slippery=True,render_mode='ansi',success_rate=0.8)

pi,V=value_iteration(env)
print('optimal policy:\n',np.array(pi).reshape([4,4]))
print('optimal value function:\n',np.round(V.reshape([4,4]),4))
