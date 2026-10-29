import gymnasium as gym
import numpy as np
import scipy
import tensorflow as tf
from tensorflow.keras.layers import Dense,Lambda
import matplotlib.pyplot as plt
import tqdm

def gaussian(x,m,s): # m=mean, s=standard deviation
    return (1.0/(s*tf.math.sqrt(2.0*np.pi)))*tf.math.exp(-((x-m)**2)/(2*s**2))

def log_1(x): return tf.math.log(1+tf.math.exp(x))

def neural_network(): # make network
    input_x=tf.keras.Input(shape=(s_dim,))
    hidden_x=Dense(32,activation='tanh')(input_x)
    hidden_x=Dense(32,activation='tanh')(hidden_x)
    mean=Dense(a_dim,activation='linear')(hidden_x)
    std1=Dense(a_dim,activation='linear')(hidden_x)
    std=Lambda(log_1,output_shape=(a_dim,))(std1)
    mlp=tf.keras.Model(inputs=input_x,outputs=[mean,std])
    return mlp

def discount_cumulation(x,discount):
    return scipy.signal.lfilter([1],[1,float(-discount)],x[::-1],axis=0)[::-1]

def learn(): # train network using episode e
    return_mem=[]
    return_mem=discount_cumulation(reward_mem,gamma) # returns
    return_mem=tf.convert_to_tensor(return_mem,dtype=tf.float32)

    with tf.GradientTape() as tape:
        mean,std=model(tf.convert_to_tensor(state_mem))
        logprob=tf.math.log(gaussian(action_mem,mean,std))
        loss=-tf.reduce_sum(tf.squeeze(logprob)*return_mem)
    gradient=tape.gradient(loss,model.trainable_weights)
    optimizer.apply_gradients(zip(gradient,model.trainable_weights))

gamma=0.99 # discount rate
n_episode=1000

env=gym.make('InvertedPendulum-v4',render_mode='rgb_array')
s_dim=env.observation_space.shape[0] # state space
a_dim=env.action_space.shape[0] # action space
a_low,a_high=env.action_space.low,env.action_space.high

model=neural_network() # make network
optimizer=tf.optimizers.Adam(learning_rate=0.001)

state_mem=[] # memory for storing episode data
action_mem=[]
reward_mem=[]

epi_length=[] # lengths of episodes
for e in tqdm.tqdm(range(n_episode)): # train network
    s,info=env.reset()
    while True: # episode generation
        mean,std=model(s.reshape([1,s_dim]))
        a=np.clip(np.random.normal(mean[0],std[0],1),a_low[0],a_high[0])
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

    if (e+1)%10==0: print(e+1,'-th episode average length:',np.mean(epi_length[-10:]))
    if np.min(epi_length[-5:])>=env.spec.max_episode_steps: # convergence test
        break

model.save('f8-6.keras') # store the network
env.close()

plt.figure(figsize=(16,5))
plt.plot(range(1,len(epi_length)+1),epi_length)
smooth=np.convolve(epi_length,10*[0.1],mode='valid')
plt.plot(range(1,len(smooth)+1),smooth)
plt.title('REINFORCE scores for InvertedPendulum-v4')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.grid()
plt.show()
