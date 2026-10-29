import gymnasium as gym
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import ale_py
from collections import deque
import cv2 as cv
import time

gamma=0.99 # discount rate
eps=1.0
eps_decay=0.99999
eps_min=0.01
replay_memory_siz=20000
batch_siz=32
copy_interval=1000 # period copying behavior network to target network
n_match=500
consecutive=4 # state represented by 4 consecutive frames
rows=84 # input image size for network
cols=84
actions=[0,2,3] # 0(stop), 2(up), 3(down)

greedy_select=lambda x:np.random.choice(np.argwhere(x==np.max(x)).flatten())

def deep_network():
    cnn=Sequential()
    cnn.add(Conv2D(32,(5,5),activation='relu',strides=2,input_shape=(rows,cols,consecutive)))
    cnn.add(Conv2D(64,(3,3),activation='relu',strides=2))
    cnn.add(Conv2D(64,(3,3),activation='relu',strides=1))
    cnn.add(Flatten())
    cnn.add(Dense(512,activation='relu'))
    cnn.add(Dense(len(actions),activation='linear'))
    cnn.compile(loss='MSE',optimizer=Adam(learning_rate=0.00005))
    return cnn

def preprocess(img): # image preprocessing(260x160x3 -> 84x84x1)
    crop=cv.cvtColor(img[30:-12,5:-4,:],cv.COLOR_BGR2GRAY) # remove the boundaries
    return cv.resize(crop,dsize=(rows,cols)).reshape(rows,cols,1)

def model_learning():
    mini_batch=random.sample(range(len(R)),batch_siz)
    s=np.asarray([R[mini_batch[i]][0] for i in range(batch_siz)])
    a=np.asarray([R[mini_batch[i]][1] for i in range(batch_siz)])
    r=np.asarray([R[mini_batch[i]][2] for i in range(batch_siz)])
    s1=np.asarray([R[mini_batch[i]][3] for i in range(batch_siz)])
    terminated=np.asarray([R[mini_batch[i]][4] for i in range(batch_siz)])

    X=s
    Y=np.array(model_behavior(s))
    y1_=np.array(model_target(s1))

    for i in range(batch_siz):
        if terminated[i]:
            Y[i,a[i]]=r[i]
        else:
            Y[i,a[i]]=r[i]+gamma*np.max(y1_[i])

    model_behavior.fit(X,Y,batch_size=batch_siz,epochs=1,verbose=0)

env=gym.make('ALE/Pong-v5',render_mode='rgb_array')

model_target=deep_network() # target network
model_behavior=deep_network() # behavior network

R=deque(maxlen=replay_memory_siz) # replay memory initialization
n_update=0 # number of frame accumulation
scores=[] # store the match score
start_time_total=time.time()

for i in range(n_match): # repeating matches
    start=time.time()
    obs,info=env.reset()
    imem=deque(maxlen=consecutive) # state represented by 4 consecutive frames
    for j in range(consecutive): imem.append(preprocess(obs))
    s=np.transpose(np.squeeze(np.asarray(imem)),(1,2,0))

    score=0
    while True:
        if(np.random.random()<eps):
            a=np.random.choice(len(actions))
        else:
            y_=model_behavior(s.reshape(1,rows,cols,consecutive))[0]
            a=greedy_select(y_)
        obs1,r,terminated,truncated,info=env.step(actions[a])
        imem.append(preprocess(obs1))
        s1=np.transpose(np.squeeze(np.asarray(imem)),(1,2,0))
        R.append((s,a,r,s1,terminated))
        eps=max(eps_min,eps*eps_decay)

        if len(R)>batch_siz*3 and n_update%4==0:
            model_learning()

        n_update+=1
        if n_update%copy_interval==0:
            model_target.set_weights(model_behavior.get_weights())

        s=s1

        if r!=0: # end of game
            score+=r
        if terminated or truncated: # end of match
            scores.append(score)
            break

    print(i+1,'-th match score=',scores[-1],'(',int(time.time()-start),'seconds) eps=',np.round(eps,8))

print('total time for learning=',int(time.time()-start_time_total),'seconds')
model_target.save('f7-5.keras') # store the network
env.close()

plt.plot(range(1,len(scores)+1),scores)
smooth=np.convolve(scores,10*[0.1],mode='valid')
plt.plot(range(1,len(smooth)+1),smooth)
plt.title('DQN scores for Pong-v5')
plt.ylabel('Score')
plt.xlabel('Match')
plt.grid()
plt.show()
