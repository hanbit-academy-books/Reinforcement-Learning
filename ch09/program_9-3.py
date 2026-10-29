import gymnasium as gym
import numpy as np
import scipy
import tensorflow as tf
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

def neural_network_actor_critic(): # actor-critic shared network
    input_x=tf.keras.Input(shape=(s_dim,))
    hidden_x=Dense(32,activation='tanh')(input_x)
    actor_y=Dense(a_dim,activation='softmax')(hidden_x)
    critic_y=Dense(1,activation='linear')(hidden_x)
    mlp=tf.keras.Model(inputs=input_x,outputs=[actor_y,critic_y])
    return mlp

def discount_cumulation(x,discount):
    return scipy.signal.lfilter([1],[1,-discount],x[::-1],axis=0)[::-1]

def advantage_MC(rewards,values,gamma): # Monte Carlo advantage
    return discount_cumulation(rewards,gamma)-np.array(values)

actionprob_mem=[] # data storage
value_mem=[]
reward_mem=[]

def episode_generate_and_learn():
    s,info=env.reset()
    with tf.GradientTape (persistent=True) as tape:
        while True:
            action_prob,value=ac_model(s.reshape([1,s_dim]))
            a=np.random.choice(a_dim,p=np.array(action_prob[0]))
            s1,r,terminated,truncated,info=env.step(a)

            actionprob_mem.append(action_prob[0,a])
            value_mem.append(value[0][0])
            reward_mem.append(r)
            s=s1

            if terminated or truncated:
                break

        returns=discount_cumulation(reward_mem,gamma)
        advantage=advantage_MC(reward_mem,value_mem,gamma)
        advantage=tf.convert_to_tensor(advantage,dtype=tf.float32)
        returns=tf.convert_to_tensor(returns,dtype=tf.float32)

        loss_actor=-advantage*tf.math.log(actionprob_mem)
        loss_critic=tf.math.square(returns-value_mem)
        loss=loss_actor+loss_critic

        grad=tape.gradient(tf.reduce_mean(loss),ac_model.trainable_weights)
        optimizer.apply_gradients(zip(grad,ac_model.trainable_weights))

gamma=0.99
n_episode=1000

env=gym.make('CartPole-v1')
s_dim=env.observation_space.shape[0]
a_dim=env.action_space.n

ac_model=neural_network_actor_critic() # make actor-critic network
optimizer=tf.optimizers.Adam(learning_rate=0.01)

epi_length=[]
for i in range(n_episode):
    episode_generate_and_learn()
    epi_length.append(len(actionprob_mem))
    actionprob_mem,reward_mem,value_mem=[],[],[] # make data storage empty

    if (i+1)%20==0: print(i+1,'-th episodes average length:',np.mean(epi_length[-20:]))
    if np.min(epi_length[-5:])>=env.spec.max_episode_steps: # convergence test
        break

ac_model.save('f9-3.keras')
env.close()

plt.figure(figsize=(16,5))
plt.plot(range(1,len(epi_length)+1),epi_length)
smooth=np.convolve(epi_length,10*[0.1],mode='valid')
plt.plot(range(1,len(smooth)+1),smooth)
plt.title('A2C scores for CartPole-v1 using shared network')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.grid()
plt.show()
