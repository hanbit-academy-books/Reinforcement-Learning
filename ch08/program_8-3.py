import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

gamma=0.99 # discount rate
n_episode=1000 # number of episodes

def neural_network(): # make MLP
    mlp=Sequential()
    mlp.add(Dense(32,input_dim=s_dim,activation='relu'))
    mlp.add(Dense(a_dim,activation='softmax'))
    return mlp

def learn():
    return_mem=[]
    ret=0
    for t in reversed(range(len(action_mem))): # return calculation for each time step
        ret=gamma*ret+reward_mem[t]
        return_mem.insert(0,ret)

    with tf.GradientTape() as tape:
        loss=0
        for t in range(len(action_mem)):
            y_=model(state_mem[t].reshape([1,s_dim]))[0]
            logprob=tf.math.log(y_[action_mem[t]])
            loss-=logprob*return_mem[t] # converting maximization to minimization by negation
    grad=tape.gradient(loss,model.trainable_weights)
    optimizer.apply_gradients(zip(grad,model.trainable_weights))

env=gym.make('CartPole-v1')
s_dim=env.observation_space.shape[0]
a_dim=env.action_space.n

model=neural_network() # make policy network
optimizer=tf.optimizers.Adam(learning_rate=0.001)

state_mem=[] # memory for storing episode data
action_mem=[]
reward_mem=[]

epi_length=[] # episodes length
for e in range(n_episode): # network training
    s,info=env.reset()
    while True: # episode generation
        y_=model(s.reshape([1,s_dim]))[0]
        a=np.random.choice(a_dim,p=np.array(y_))
        s1,r,terminated,truncated,info=env.step(a)
        state_mem.append(s)
        action_mem.append(a)
        reward_mem.append(r)
        s=s1
        if terminated or truncated:
            epi_length.append(len(action_mem))
            break

    learn()

    state_mem,action_mem,reward_mem=[],[],[] # make empty

    print(e,'-th episode length:',epi_length[-1])
    if np.min(epi_length[-5:])>=env.spec.max_episode_steps: # early stopping
        break

model.save('f8-2.keras') # save the network
env.close()

plt.plot(range(1,len(epi_length)+1),epi_length)
smooth=np.convolve(epi_length,10*[0.1],mode='valid')
plt.plot(range(1,len(smooth)+1),smooth)
plt.title('REINFORCE scores for CartPole-v1')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.grid()
plt.show()
