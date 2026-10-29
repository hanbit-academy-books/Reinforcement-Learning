import gymnasium as gym
import numpy as np
from tensorflow.keras.models import load_model
import cv2 as cv

model=load_model('./f6-2.keras') # load the neural network

env=gym.make('CartPole-v1',render_mode='rgb_array')
s_dim=env.observation_space.shape[0] # number of state variables

length=0
s,info=env.reset()
while True:
    q=model.predict(np.reshape(s,[1,s_dim]),verbose=0) # action values predicted by the neural network
    a=np.argmax(q[0])
    s,r,terminated,truncated,info=env.step(a)
    length+=1

    cv.imshow('CartPole animation',cv.cvtColor(env.render(),cv.COLOR_BGR2RGB))
    key=cv.waitKey(100)

    if terminated or truncated:
        print('score of the episode:',length)
        break

env.close()

if cv.waitKey()==ord('q'):
    cv.destroyAllWindows()
