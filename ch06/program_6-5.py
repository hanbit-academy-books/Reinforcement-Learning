import gymnasium as gym
import numpy as np
from tensorflow.keras.models import load_model
import tqdm

n_game=10000 # number of games

greedy_select=lambda x:np.random.choice(np.argwhere(x==np.max(x)).flatten())

env=gym.make('Blackjack-v1', render_mode='rgb_array')
s_dim=len(env.observation_space)

q=load_model('f6-4.keras')
n_win,n_lose,n_draw=0,0,0
for i in tqdm.tqdm(range(n_game)):
    s,info=env.reset()
    while True:
        y_=q.predict(np.array(s).reshape([1,s_dim]),verbose=0)[0]
        a=greedy_select(y_)
        s,r,terminated,truncated,info=env.step(a)
        if terminated or truncated:
            if r==1: n_win+=1
            elif r==-1: n_lose+=1
            elif r==0: n_draw+=1
            break
print('\n win=',n_win,'loss=',n_lose,'draw=',n_draw)

env.close()
