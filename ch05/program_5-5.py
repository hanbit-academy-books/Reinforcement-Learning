import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

greedy_select=lambda x:np.random.choice(np.argwhere(x==np.max(x)).flatten())
def init_q(): return np.zeros(env.action_space.n)

env=gym.make('Blackjack-v1', render_mode='rgb_array')
Q=pickle.load(open('f5-4.pkl','rb'))

s,info=env.reset()
print('initial state=',s)
plt.imshow(env.render())
plt.show()
while True:
    a=greedy_select(Q[s])
    s1,r,terminated,truncated,info=env.step(a)

    print('action=',a,'state=',s1,'reward=',r,'end?=',terminated)
    plt.imshow(env.render())
    plt.show()

    s=s1
    if terminated or truncated:
        break

env.close()
