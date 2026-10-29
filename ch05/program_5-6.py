import gymnasium as gym
import numpy as np
import pickle

n_game=10000 # number of games

greedy_select=lambda x:np.random.choice(np.argwhere(x==np.max(x)).flatten())
def init_q(): return np.zeros(env.action_space.n)

env=gym.make('Blackjack-v1', render_mode='rgb_array')

n_win,n_lose,n_draw=0,0,0 # profit analysis for random policy
for i in range(n_game):
    s,info=env.reset()
    while True:
        a=np.random.choice([0,1],p=(0.5,0.5))
        s1,r,terminated,truncated,info=env.step(a)
        s=s1
        if terminated or truncated:
            if r==1: n_win+=1
            elif r==-1: n_lose+=1
            elif r==0: n_draw+=1
            break
print('random policy: wins=',n_win,'losses=',n_lose,'ties=',n_draw)

Q=pickle.load(open('f5-4.pkl','rb')) # profit analysis for Q-learning
n_win,n_lose,n_draw=0,0,0
for i in range(n_game):
    s,info=env.reset()
    while True:
        a=greedy_select(Q[s])
        s1,r,terminated,truncated,info=env.step(a)
        s=s1
        if terminated or truncated:
            if r==1: n_win+=1
            elif r==-1: n_lose+=1
            elif r==0: n_draw+=1
            break
print('Q-learning: wins=',n_win,'losses=',n_lose,'ties=',n_draw)

env.close()
