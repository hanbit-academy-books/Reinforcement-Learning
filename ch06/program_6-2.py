import numpy as np
import gymnasium as gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

gamma=0.99 # discount rate
eps=1.0
eps_decay=0.9999
eps_min=0.05
n_episode=2000

greedy_select=lambda x:np.random.choice(np.argwhere(x==np.max(x)).flatten())

def build_network(): # making a neural network(MLP)
    mlp=Sequential()
    mlp.add(Dense(32,input_dim=s_dim,activation='relu'))
    mlp.add(Dense(32,activation='relu'))
    mlp.add(Dense(a_dim,activation='linear'))
    mlp.compile(loss='MSE',optimizer=Adam(learning_rate=0.01))
    return mlp

env=gym.make('CartPole-v1')
s_dim=env.observation_space.shape[0] # number of state variables
a_dim=env.action_space.n # number of action variables

model=build_network() # make a neural network

epi_length=[] # episode length
for i in range(n_episode): # training neural network
    s,info=env.reset()
    length=0
    while True:
        y_=model.predict(s.reshape([1,s_dim]),verbose=0)[0]
        if np.random.random()<eps:
            a=np.random.randint(0,a_dim)
        else:
            a=greedy_select(y_)
        s1,r,terminated,truncated,info=env.step(a)
        x=s; y=y_
        y1_=model.predict(s1.reshape([1,s_dim]),verbose=0)[0]

        if terminated:
            y[a]=r
        else:
            y[a]=r+gamma*np.max(y1_)

        model.fit(x.reshape([1,s_dim]),y.reshape([1,a_dim]),batch_size=1,epochs=1,verbose=0)

        s=s1
        length+=1
        eps=max(eps_min,eps*eps_decay) # epsilon scheduling

        if terminated or truncated:
            epi_length.append(length)
            break

    if np.min(epi_length[-5:])>=env.spec.max_episode_steps: # convergence if exceeding max length over 5 consecutive runs
        break
    if (i+1)%10==0: print(i+1,'-th episode length:',np.mean(epi_length[-10:]))

model.save('./f6-2.keras') # save the neural network
env.close()

plt.plot(range(1,len(epi_length)+1),epi_length) # drawing learning curve
smooth=np.convolve(epi_length,10*[0.1],mode='valid')
plt.plot(range(1,len(smooth)+1),smooth)
plt.title('Convergence of Q-network for CartPole-v1')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.grid()
plt.show()
