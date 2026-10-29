import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env=gym.make('Blackjack-v1', render_mode='rgb_array')
print(env.observation_space,env.action_space)

s,info=env.reset()
print('state=',s)
plt.imshow(env.render())
plt.show()
while True:
    a=np.random.choice([0,1],p=(0.5,0.5))
    s1,r,terminated,truncated,info=env.step(a)

    print('action=',a,'state=',s1,'reward=',r,'end?=',terminated)
    plt.imshow(env.render())
    plt.show()

    if terminated or truncated:
        break

env.close()
