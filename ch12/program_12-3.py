import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt

class ActorNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.actor_mean=tf.keras.Sequential(
            [Input(shape=(s_dim,)),
             Dense(64,activation=activation,kernel_initializer=initializer),
             Dense(64,activation=activation,kernel_initializer=initializer),
             Dense(a_dim,kernel_initializer=initializer)])

    def call(self,state):
        mean=self.actor_mean(state)
        return mean

activation=tf.keras.activations.tanh
initializer=tf.keras.initializers.Orthogonal

def expert_surrogator(): # policy network trained by [Program 10-1] surrogating the expert
    actor=ActorNetwork()
    actor.load_weights('f10-1.weights.h5')
    return actor

def mlp_network(): # network to be trained by BC
    nn=Sequential()
    nn.add(Dense(64,input_shape=(s_dim,)))
    nn.add(Dense(64))
    nn.add(Dense(a_dim))
    return nn

def state_data_by_learner(policy): # state seq collection by learner
    state_traj=[]
    s,info=env.reset()
    while True:
        mean=policy(s.reshape(1,-1))
        action=mean[0]
        s1,r,terminated,truncated,info=env.step(action)
        state_traj.append(s)
        s=s1
        if terminated or truncated:
            break
    return state_traj

def action_labeling_by_expert(policy,states): # action labeling by expert
    actions=[]
    for s in states:
        actions.append(policy(s.reshape(1,-1))[0])
    return actions

env=gym.make('Hopper-v4',render_mode='rgb_array')
s_dim=env.observation_space.shape[0]
a_dim=env.action_space.shape[0]

expert=expert_surrogator() # network surrogating the expert
learner=mlp_network() # learner network
learner.compile(loss='MSE',optimizer=Adam())

Xbuffer,Ybuffer=deque(maxlen=10000),deque(maxlen=10000) # using deque to store the recent data
epi_length=[]
for i in range(300):
    X=state_data_by_learner(learner)
    Y=action_labeling_by_expert(expert,X)
    epi_length.append(len(X))
    Xbuffer.extend(X)
    Ybuffer.extend(Y)
    hist=learner.fit(np.array(Xbuffer),np.array(Ybuffer),batch_size=128,epochs=1,verbose=0)
    if i%20==0: print('epoch',i,' loss=',np.mean(hist.history['loss']),'trajectory length=',len(X))

learner.save('f12-3.keras') # storing the network in file
env.close()

plt.figure(figsize=(16,5))
plt.plot(range(1,len(epi_length)+1),epi_length)
smooth=np.convolve(epi_length,10*[0.1],mode='valid')
plt.plot(range(1,len(smooth)+1),smooth)
plt.title('DAgger trajectory lengths for Hopper-v4 using shared network')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.grid()
plt.show()
