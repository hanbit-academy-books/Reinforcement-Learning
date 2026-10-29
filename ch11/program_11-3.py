import gymnasium as gym
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,LayerNormalization,Activation,Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomUniform
import matplotlib.pyplot as plt
import tqdm

class ReplayBuffer:
    def __init__(self,max_siz):
        self.state=deque(maxlen=max_siz) # state
        self.action=deque(maxlen=max_siz) # action
        self.reward=deque(maxlen=max_siz) # reward
        self.state1=deque(maxlen=max_siz) # next state
        self.done=deque(maxlen=max_siz) # episode termination

    def store(self,s,a,r,s1,d):
        self.state.append(s)
        self.action.append(a)
        self.reward.append(r)
        self.state1.append(s1)
        self.done.append(d)

    def get(self,siz):
        idx=np.random.choice(len(self.state),siz)
        s=tf.convert_to_tensor([self.state[i] for i in idx],dtype=tf.float32)
        a=tf.convert_to_tensor([self.action[i] for i in idx],dtype=tf.float32)
        r=tf.convert_to_tensor([self.reward[i] for i in idx],dtype=tf.float32)
        s1=tf.convert_to_tensor([self.state1[i] for i in idx],dtype=tf.float32)
        d=tf.convert_to_tensor([1.0 if self.done[i] else 0.0 for i in idx],dtype=tf.float32)
        return s,a,r,s1,d

def build_actor():
    inputs=Input(shape=(s_dim,))
    x=Dense(256)(inputs)
    x=LayerNormalization()(x)
    x=Activation('relu')(x)
    x=Dense(256)(x)
    x=LayerNormalization()(x)
    x=Activation('relu')(x)
    outputs=Dense(a_dim,activation='tanh',kernel_initializer=init)(x)
    outputs=outputs*action_bound
    model=Model(inputs,outputs)
    return model

def build_critic():
    state_input=Input(shape=(s_dim,)) # branch for state
    action_input=Input(shape=(a_dim,)) # branch for action
    x=Dense(256)(state_input)
    x=LayerNormalization()(x)
    x=Activation('relu')(x)
    concat=Concatenate()([x,action_input])
    x=Dense(256)(concat)
    x=LayerNormalization()(x)
    x=Activation('relu')(x)
    outputs=Dense(1,kernel_initializer=init)(x)
    model=Model([state_input,action_input],outputs)
    return model

@tf.function
def learn_critic(states,actions,rewards,next_states,dones):
    with tf.GradientTape(persistent=True) as tape:
        noise=tf.clip_by_value(tf.random.normal(shape=(batch_siz,a_dim),stddev=0.2),-0.5,0.5)
        next_action=tf.clip_by_value(actor_target(next_states)+noise,-action_bound,action_bound)
        q1_target=tf.squeeze(critic_target1([next_states,next_action]),1)
        q2_target=tf.squeeze(critic_target2([next_states,next_action]),1)
        y=rewards+gamma*tf.minimum(q1_target,q2_target)*(1-dones)
        q1=tf.squeeze(critic1([states,actions]),1)
        q2=tf.squeeze(critic2([states,actions]),1)
        loss1=tf.reduce_mean(tf.square(y-q1))
        loss2=tf.reduce_mean(tf.square(y-q2))
    grad1=tape.gradient(loss1,critic1.trainable_variables)
    grad2=tape.gradient(loss2,critic2.trainable_variables)
    optimizer_critic1.apply_gradients(zip(grad1,critic1.trainable_variables))
    optimizer_critic2.apply_gradients(zip(grad2,critic2.trainable_variables))

@tf.function
def learn_actor(states):
    with tf.GradientTape() as tape:
        actions=actor(states)
        q=critic1([states,actions])
        loss=-tf.reduce_mean(q)
    grad=tape.gradient(loss,actor.trainable_variables)
    optimizer_actor.apply_gradients(zip(grad,actor.trainable_variables))

def update_target_networks():
    def update_weights(source,target):
        for i in range(len(source)):
            target[i]=polyak*target[i]+(1-polyak)*source[i]
        return target
    actor_target.set_weights(update_weights(actor.get_weights(),actor_target.get_weights()))
    critic_target1.set_weights(update_weights(critic1.get_weights(),critic_target1.get_weights()))
    critic_target2.set_weights(update_weights(critic2.get_weights(),critic_target2.get_weights()))

def sample_action(state): # action sampling
    action=actor(state)+np.random.normal(0,eps_scale,size=a_dim)
    return tf.clip_by_value(action,-action_bound,action_bound)

gamma=0.99 # discount factor
buffer_max_siz=1000000 # replay memory size
batch_siz=128 # minibatch size
n_episode=3000 # number of iterating episodes
eps_scale,eps_decay,eps_min=1.0,0.999,0.01 # noise scheduling
learn_period,n_learn,policy_delay=1,1,1 # learning cycle and delay
actor_lr,critic_lr=1e-4,1e-3 # learning rate
polyak=0.995 # Polyak rate
init=RandomUniform(minval=-0.003,maxval=0.003) # initialization of output layer

task_name='Hopper-v4'
env=gym.make(task_name)
s_dim=env.observation_space.shape[0]
a_dim=env.action_space.shape[0]
action_bound=env.action_space.high[0]

actor,actor_target=build_actor(),build_actor()
critic1,critic_target1=build_critic(),build_critic()
critic2,critic_target2=build_critic(),build_critic()
actor_target.set_weights(actor.get_weights())
critic_target1.set_weights(critic1.get_weights())
critic_target2.set_weights(critic2.get_weights())

optimizer_actor=tf.keras.optimizers.Adam(learning_rate=actor_lr)
optimizer_critic1=tf.keras.optimizers.Adam(learning_rate=critic_lr)
optimizer_critic2=tf.keras.optimizers.Adam(learning_rate=critic_lr)

mem=ReplayBuffer(buffer_max_siz)

epi_length,epi_score,n_steps=[],[],[0]
for e in tqdm.tqdm(range(n_episode)):
    steps,score=0,0
    state,_=env.reset()
    while True:
        action=sample_action(state.reshape(1,-1))[0]
        state1,reward,terminated,truncated,_=env.step(action)
        mem.store(state,action,reward,state1,terminated)
        state=state1

        eps_scale=max(eps_min,eps_scale*eps_decay)
        score+=reward
        steps+=1

        if len(mem.state)>batch_siz and steps%learn_period==0: # learning
            for j in range(n_learn):
                s_batch,a_batch,r_batch,s1_batch,d_batch=mem.get(batch_siz)
                learn_critic(s_batch,a_batch,r_batch,s1_batch,d_batch)
                if j%policy_delay==0: # delay of policy network learning
                    learn_actor(s_batch)
                    update_target_networks()

        if terminated or truncated:
                n_steps.append(n_steps[-1]+steps)
                epi_length.append(steps)
                epi_score.append(score)
                break

    mlen,mscore=np.mean(epi_length[-20:]),np.mean(epi_score[-20:])
    if (len(epi_length)+1)%20==0:
        print(f' length and score of episodes (total time steps) {mlen:.2f},{mscore:.2f}({n_steps[-1]})')

actor.save('f11-3.keras')
env.close()

plt.figure(figsize=(16,5))
smooth=np.convolve(epi_length,10*[0.1],mode='valid')
plt.plot(smooth,label='Length')
smooth=np.convolve(epi_score,10*[0.1],mode='valid')
plt.plot(smooth,label='Score')
plt.legend()
plt.grid()
plt.show()
