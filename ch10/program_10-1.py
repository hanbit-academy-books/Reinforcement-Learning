import gymnasium as gym
import numpy as np
import scipy
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense
import matplotlib.pyplot as plt
import tqdm

def discount_cumulation(x,discount):
    return scipy.signal.lfilter([1],[1,float(-discount)],x[::-1],axis=0)[::-1]

class RolloutBuffer:
    def __init__(self,siz):
        self.state_mem=np.zeros((siz,s_dim),dtype=np.float32) # state
        self.action_mem=np.zeros((siz,a_dim),dtype=np.float32) # action
        self.advantage_mem=np.zeros(siz,dtype=np.float32) # advantage
        self.reward_mem=np.zeros(siz,dtype=np.float32) # reward
        self.return_mem=np.zeros(siz,dtype=np.float32) # return
        self.value_mem=np.zeros(siz,dtype=np.float32) # value
        self.logprob_mem=np.zeros(siz,dtype=np.float32) # log probability of action
        self.t_start,self.t_end=0,0

    def store(self,state,action,reward,value,logprob):
        self.state_mem[self.t_end]=state
        self.action_mem[self.t_end]=action
        self.reward_mem[self.t_end]=reward
        self.value_mem[self.t_end]=value
        self.logprob_mem[self.t_end]=logprob
        self.t_end+=1

    def finish_trajectory(self,last_value=0):
        tslice=slice(self.t_start,self.t_end)
        rewards=np.append(self.reward_mem[tslice],last_value)
        values=np.append(self.value_mem[tslice],last_value)

        deltas=rewards[:-1]+gamma*values[1:]-values[:-1]
        self.advantage_mem[tslice]=discount_cumulation(deltas,gamma*lamda) # GAE
        self.return_mem[tslice]=discount_cumulation(rewards,gamma)[:-1] # return
        self.t_start=self.t_end

    def get(self):
        self.t_start,self.t_end=0,0
        if advantage_normalizing: # applying advantage normalization
            mean,std=np.mean(self.advantage_mem),np.std(self.advantage_mem)
            self.advantage_mem=(self.advantage_mem-mean)/std
        return (self.state_mem,self.action_mem,self.advantage_mem,self.return_mem,self.logprob_mem)

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

    def sample_action(self,state,std): # sampling action
        mean=self.call(state)
        noise=tf.random.normal(shape=mean.shape)
        action=mean+noise*std
        return action

    def log_prob(self,state,action,std): # log of Gaussian
        mean=self.call(state)
        log_v1=-0.5*tf.square((action-mean)/std)
        log_v2=0.5*tf.math.log(2.0*np.pi)+tf.math.log(std)
        return tf.reduce_sum(log_v1-log_v2,axis=1)

class CriticNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.critic=tf.keras.Sequential(
            [Input(shape=(s_dim,)),
             Dense(64,activation=activation,kernel_initializer=initializer),
             Dense(64,activation=activation,kernel_initializer=initializer),
             Dense(1,kernel_initializer=initializer)])

    def call(self,state):
        return tf.keras.ops.squeeze(self.critic(state),axis=1)

@tf.function
def learn_actor(states,actions,logprobs,advantages,std):
    with tf.GradientTape() as tape:
        logprobs_new=actor.log_prob(states,actions,std)
        ratio=tf.exp(logprobs_new-logprobs)
        mins=tf.where(advantages>0,(1+clip_epsilon)*advantages,(1-clip_epsilon)*advantages)
        loss=-tf.reduce_mean(tf.minimum(ratio*advantages,mins))
        if entropy_loss_term: # applying entropy
            entropy=tf.reduce_mean(tf.math.log(std)+0.5*tf.math.log(2.0*np.pi*np.e))
            entropy_loss=0.01*entropy
            loss+=entropy_loss
    grad=tape.gradient(loss,actor.trainable_variables)
    optimizer_actor.apply_gradients(zip(grad,actor.trainable_variables))
    kl=tf.reduce_mean(logprobs-logprobs_new) # used by early stopping
    return kl

@tf.function
def learn_critic(states,returns):
    with tf.GradientTape() as tape:
        values=critic(states)
        loss=tf.reduce_mean((returns-values)**2)
        grad=tape.gradient(loss,critic.trainable_variables)
        optimizer_critic.apply_gradients(zip(grad,critic.trainable_variables))

gamma,lamda=0.99,0.95 # discount rate, GAE lambda
steps_per_epoch,n_epoch=4096,1000 # rollout buffer size, number of epochs
actor_lr,critic_lr=3e-4,1e-3 # learning rates
actor_iterations,critic_iterations=80,80 # iterations per epoch
clip_epsilon=0.2 # epsilon for PPO clip
target_kl=0.01 # early stopping
activation=tf.keras.activations.tanh # activation functions
std,std_decay,std_min=0.6065,0.999,0.01 # std scheduling
initializer=tf.keras.initializers.Orthogonal # weights initialization
advantage_normalizing=False # whether to apply advantage normalization
actor_early_stopping=False # whether to apply early stopping
entropy_loss_term=False # whether to apply entropy

task_name='Hopper-v4'
env=gym.make(task_name)
s_dim=env.observation_space.shape[0]
a_dim=env.action_space.shape[0]

mem=RolloutBuffer(steps_per_epoch)
actor=ActorNetwork()
critic=CriticNetwork()
optimizer_actor=tf.keras.optimizers.Adam(learning_rate=actor_lr)
optimizer_critic=tf.keras.optimizers.Adam(learning_rate=critic_lr)

epi_lengths,epi_scores=[],[]
length_per_epoch,score_per_epoch=[],[]
state,info=env.reset()
for epoch in tqdm.tqdm(range(n_epoch)):
    n_epi,length,score=0,0,0
    std=max(std_min,std*std_decay) # std scheduling
    for t in range(steps_per_epoch):
        action=actor.sample_action(state.reshape(1,-1),std)
        logprob_t=actor.log_prob(state.reshape(1,-1),action,std)
        value_t=critic(state.reshape(1,-1))
        state_,reward,terminated,truncated,_=env.step(action[0].numpy())
        mem.store(state,action[0],reward,value_t,logprob_t)
        state=state_

        score+=reward
        length+=1

        if (terminated or truncated) or (t==steps_per_epoch-1):
            last_value=0 if (terminated or truncated) else critic(state.reshape(1,-1))
            mem.finish_trajectory(last_value)

            if terminated or truncated:
                n_epi+=1
            epi_lengths.append(length)
            epi_scores.append(score)

        length,score=0,0
        state,info=env.reset()

    states,actions,advantages,returns,logprobs=mem.get()
    for k in range(actor_iterations): # training actor
        kl=learn_actor(states,actions,logprobs,advantages,std)
        if actor_early_stopping and kl>1.5*target_kl: # early stopping
            break
    for k in range(critic_iterations): # training critic
        learn_critic(states,returns)

    length_per_epoch.append(np.mean(epi_lengths[-n_epi:]))
    score_per_epoch.append(np.mean(epi_scores[-n_epi:]))
    if (epoch+1)%10==0:
        print(f' episode length(score) {np.mean(length_per_epoch[-10:]):.2f},{np.mean(score_per_epoch[-10:]):.2f}')

actor.save_weights('f10-1.weights.h5')
env.close()

plt.figure(figsize=(16,5))
smooth=np.convolve(score_per_epoch,10*[0.1],mode='valid')
plt.plot(range(1,len(smooth)+1),smooth)
smooth=np.convolve(length_per_epoch,10*[0.1],mode='valid')
plt.plot(range(1,len(smooth)+1),smooth)
plt.title('PPO scores for '+task_name)
plt.ylabel('Length and Score')
plt.xlabel('Epoch')
plt.legend(['Score','Length'])
plt.grid()
plt.show()
