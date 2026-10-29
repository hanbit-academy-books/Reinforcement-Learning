import gymnasium as gym
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt

gamma=0.99 # discount rate
eps=1.0
eps_decay=0.999
eps_min=0.05
replay_memory_siz=2000
batch_siz=64
n_episode=500

greedy_select=lambda x:np.random.choice(np.argwhere(x==np.max(x)).flatten())

def build_network(): # making a neural network
    mlp=Sequential()
    mlp.add(Dense(32,input_dim=s_dim,activation='relu'))
    mlp.add(Dense(32,activation='relu'))
    mlp.add(Dense(a_dim,activation='linear'))
    mlp.compile(loss='MSE',optimizer=Adam(learning_rate=0.001))
    return mlp

def model_learning(): # DQN learning
    mini_batch=random.sample(R,batch_siz)
    s=np.asarray([sample[0] for sample in mini_batch])
    a=[sample[1] for sample in mini_batch]
    r=[sample[2] for sample in mini_batch]
    s1=np.asarray([sample[3] for sample in mini_batch])
    terminated=[sample[4] for sample in mini_batch]

    X=s
    Y=model.predict(s,verbose=0)
    y1_=model.predict(s1,verbose=0)

    for i in range(batch_siz):
        if terminated[i]:
            Y[i,a[i]]=r[i]
        else:
            Y[i,a[i]]=r[i]+gamma*np.max(y1_[i])

    model.fit(X,Y,batch_size=batch_siz,epochs=1,verbose=0)

env=gym.make('CartPole-v1')
s_dim=env.observation_space.shape[0] # number of state variables
a_dim=env.action_space.n # number of action values

model=build_network() # make network
R=deque(maxlen=replay_memory_siz) # replay memory initialization
epi_length=[] # storing episode lengths

for i in range(n_episode): # network training
    s,info=env.reset()
    length=0
    while True:
        y_=model.predict(s.reshape([1,s_dim]),verbose=0)[0]
        if(np.random.random()<eps):
            a=np.random.randint(0,a_dim)
        else:
            a=greedy_select(y_)
        s1,r,terminated,truncated,info=env.step(a)
        R.append((s,a,r,s1,terminated))

        if len(R)>batch_siz*3:
            model_learning()

        s=s1
        length+=1
        eps=max(eps_min,eps*eps_decay) # epsilon scheduling

        if terminated or truncated:
            epi_length.append(length)
            break

    print(i+1,'-th episode length:',epi_length[i])
    if np.min(epi_length[-5:])>=env.spec.max_episode_steps: # convergence checking
        break

model.save('./f7-1.keras') # store the network
env.close()

plt.plot(range(1,len(epi_length)+1),epi_length)
plt.title('DQN scores for CartPole-v1')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.grid()
plt.show()
