import gymnasium as gym
import numpy as np
import pickle
import cv2 as cv

greedy_select=lambda x:np.random.choice(np.argwhere(x==np.max(x)).flatten())

n_bin=21 # number of bins
state_bin=[np.linspace(-2.5,2.5,n_bin),
            np.linspace(-4.0,4.0,n_bin),
            np.linspace(-0.28,0.28,n_bin),
            np.linspace(-3.9,3.9,n_bin)]

def state_discretization(state): # converting real-valued state into bin index
        bin_index=[]
        for i in range(len(state)):
            bin_index.append(np.digitize(state[i],state_bin[i])-1)
        return tuple(bin_index)

env=gym.make('CartPole-v1',render_mode='rgb_array')
Q=pickle.load(open('f5-9.pkl','rb'))

length=0
s,info=env.reset()
s_=state_discretization(s)
while True:
    a=greedy_select(Q[s_])
    s1,r,terminated,truncated,info=env.step(a)
    s1_=state_discretization(s1)
    s_=s1_
    length+=1

    cv.imshow('CartPole animation',cv.cvtColor(env.render(),cv.COLOR_BGR2RGB))
    key=cv.waitKey(100)

    if terminated or truncated:
        break

print('episode length:',length)
env.close()

if cv.waitKey()==ord('q'):
    cv.destroyAllWindows()
