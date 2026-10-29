import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

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

def mlp_network(): # MLP that will be trained by BC
    nn=Sequential()
    nn.add(Dense(64,input_shape=(s_dim,)))
    nn.add(Dense(64))
    nn.add(Dense(a_dim))
    return nn

def demonstration_data_by_expert(expert_policy,n_traj=300): # demo data collection by the expert
    demo_data=[]
    for i in range(n_traj):
        traj=[]
        s,info=env.reset()
        while True:
            mean=expert_policy(s.reshape(1,-1))
            action=mean[0]
            s1,r,terminated,truncated,info=env.step(action)
            traj.append([s,action])
            s=s1

            if terminated or truncated:
                if (i+1)%20==0: print('demo data',i+1,'-th trajectory length=',len(traj))
                demo_data.append(traj)
                break
    return demo_data

env=gym.make('Hopper-v4',render_mode='rgb_array')
s_dim=env.observation_space.shape[0]
a_dim=env.action_space.shape[0]

expert=expert_surrogator() # network surrogating the expert
policy_bc_vanilla=mlp_network() # network to be trained by BC

D=demonstration_data_by_expert(expert) # demo data collection
X=np.array([[[a for a in b[0]] for b in c] for c in D]).reshape(-1,s_dim)
Y=np.array([[[a for a in b[1]] for b in c] for c in D]).reshape(-1,a_dim)

policy_bc_vanilla.compile(loss='MSE',optimizer=Adam())
policy_bc_vanilla.fit(X,Y,batch_size=128,epochs=100,verbose=1) # training the policy network

policy_bc_vanilla.save('f12-1.keras') # storing the network in file
env.close()
