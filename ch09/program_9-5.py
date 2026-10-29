import gymnasium as gym
import numpy as np
import scipy
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense
import matplotlib.pyplot as plt
import tqdm

def discount_cumulation(x,discount):
    return scipy.signal.lfilter([1],[1,float(-discount)],x[::-1],axis=0)[::-1]

def advantage_GAE(rewards,values,gamma,lamda): # GAE advantage
    rewards,values,deltas=np.array(rewards),np.array(values),np.zeros(len(rewards))
    deltas[:-1]=rewards[:-1]+gamma*values[1:]-values[:-1]
    deltas[-1]=rewards[-1]-values[-1]
    return discount_cumulation(deltas,gamma*lamda)

class Buffer:
    def __init__(self,siz):
        self.state_mem=np.zeros((siz,s_dim),dtype=np.float32)
        self.action_mem=np.zeros((siz,a_dim),dtype=np.float32)
        self.reward_mem=np.zeros(siz,dtype=np.float32)
        self.value_mem=np.zeros(siz,dtype=np.float32)
        self.pointer=0

    def store(self,state,action,reward,value):
        self.state_mem[self.pointer]=state
        self.action_mem[self.pointer]=action
        self.reward_mem[self.pointer]=reward
        self.value_mem[self.pointer]=value
        self.pointer+=1

    def finish_trajectory(self):
        self.advantage_mem=advantage_GAE(self.reward_mem[:self.pointer],self.value_mem[:self.pointer],gamma,lamda)
        self.return_mem=discount_cumulation(self.reward_mem[:self.pointer],gamma)

    def get(self):
        siz=self.pointer
        self.pointer=0
        return (self.state_mem[:siz],self.action_mem[:siz],self.advantage_mem[:siz],self.return_mem[:siz])

class ActorNetwork(tf.keras.Model):
    def __init__(self,):
        super().__init__()
        self.actor_mean=tf.keras.Sequential(
            [Input(shape=(s_dim,)),
             Dense(64,activation=activation,kernel_initializer=initializer),
             Dense(64,activation=activation,kernel_initializer=initializer),
             Dense(a_dim,kernel_initializer=initializer)])

    def call(self,state):
        mean=self.actor_mean(state)
        return mean

    def sample_action(self,state,std):
        mean=self.call(state)
        noise=tf.random.normal(shape=mean.shape)
        action=mean+noise*std
        return action

    def log_prob(self,action,mean,std):
        log_v1=-0.5*tf.square((action-mean)/std)
        log_v2=0.5*tf.math.log(2.0*np.pi)+tf.math.log(std)
        return tf.reduce_sum(log_v1-log_v2,axis=1)

class CriticNetwork(tf.keras.Model):
    def __init__(self,):
        super().__init__()
        self.critic=tf.keras.Sequential(
            [Input(shape=(s_dim,)),
             Dense(64,activation=activation,kernel_initializer=initializer),
             Dense(64,activation=activation,kernel_initializer=initializer),
             Dense(1,kernel_initializer=initializer)])

    def call(self,state):
        return tf.keras.ops.squeeze(self.critic(state),axis=1)

@tf.function
def learn_actor(state_,action_,advantage_,std):
    with tf.GradientTape() as tape:
        mean=actor(state_)
        logprob=actor.log_prob(action_,mean,std)
        loss=-tf.reduce_mean(logprob*advantage_)
        if entropy_loss_term: # when applying entropy
            entropy=tf.reduce_mean(tf.math.log(std)+0.5*tf.math.log(2.0*np.pi*np.e))
            entropy_loss=0.01*entropy
            loss+=entropy_loss
    grad=tape.gradient(loss,actor.trainable_variables)
    optimizer_actor.apply_gradients(zip(grad,actor.trainable_variables))

@tf.function
def learn_critic(state_,return_):
    with tf.GradientTape() as tape:
        value=critic(state_)
        loss=0.5*tf.reduce_mean(tf.square(return_-value))
    grad=tape.gradient(loss,critic.trainable_variables)
    optimizer_critic.apply_gradients(zip(grad,critic.trainable_variables))

gamma,lamda=0.99,0.95
buffer_siz,n_epoch=1000,2000
actor_lr,critic_lr=3e-4,1e-3
learn_iterations=10
activation=tf.keras.activations.tanh # activation function
initializer=tf.keras.initializers.Orthogonal # initialization of network parameters
std,std_decay=0.6065,0.999 # std annealing
entropy_loss_term=True # applying entropy

task_name='InvertedPendulum-v4'
env=gym.make(task_name)
s_dim=env.observation_space.shape[0]
a_dim=env.action_space.shape[0]

buffer=Buffer(buffer_siz)

actor=ActorNetwork()
critic=CriticNetwork()
optimizer_actor=tf.optimizers.Adam(learning_rate=actor_lr)
optimizer_critic=tf.optimizers.Adam(learning_rate=critic_lr)

epi_length,scores=[],[]
for i in tqdm.tqdm(range(n_epoch)):
    score=0
    std=std*std_decay
    s,info=env.reset()
    while True:
        action=actor.sample_action(s.reshape(1,-1),std)
        value_t=critic(s.reshape(1,-1))
        s_,r,terminated,truncated,_=env.step(action[0].numpy())

        buffer.store(s,action[0],r,value_t)
        score+=r
        s=s_

        if terminated or truncated:
            buffer.finish_trajectory()
            epi_length.append(buffer.pointer)
            scores.append(score)
            break

    state_,action_,advantage_,return_=buffer.get()
    state_=tf.convert_to_tensor(state_,dtype=tf.float32)
    action_=tf.convert_to_tensor(action_,dtype=tf.float32)
    advantage_=tf.convert_to_tensor(advantage_,dtype=tf.float32)
    return_=tf.convert_to_tensor(return_,dtype=tf.float32)

    for k in range(learn_iterations): # training actor
        learn_actor(state_,action_,advantage_,std)
        learn_critic(state_,return_)

    if (i+1)%20==0: print(i+1,'average of episode scores:',np.mean(scores[-20:]))
    if np.min(scores[-5:])>=env.spec.max_episode_steps: # convergence test
        break

actor.save_weights('f9-5.weights.h5')
env.close()

plt.figure(figsize=(16,5))
plt.plot(range(1,len(scores)+1),scores)
smooth=np.convolve(scores,10*[0.1],mode='valid')
plt.plot(range(1,len(smooth)+1),smooth)
plt.title('A2C scores for '+task_name)
plt.ylabel('Score')
plt.xlabel('Episode')
plt.grid()
plt.show()
