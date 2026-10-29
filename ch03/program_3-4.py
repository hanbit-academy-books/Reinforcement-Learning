import gymnasium as gym
import pprint # clean output of dictionary

env=gym.make('FrozenLake-v1',is_slippery=False,render_mode='ansi')
print('transition probability distribution of deterministic MDP:')
pprint.pprint(env.unwrapped.P)

env=gym.make('FrozenLake-v1',is_slippery=True,render_mode='ansi')
print('transition probability distribution of stochastic MDP:')
pprint.pprint(env.unwrapped.P)
