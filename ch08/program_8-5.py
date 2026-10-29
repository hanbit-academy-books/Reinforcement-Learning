import gymnasium as gym
import numpy as np
from tensorflow.keras.models import load_model
import cv2 as cv

model=load_model('f8-2.keras')

env=gym.make('CartPole-v1',render_mode='rgb_array')
s_dim=env.observation_space.shape[0] # number of state variables

length=0
s,info=env.reset()
while True:
    action_prob=model(s.reshape([1,s_dim]))
    a=np.argmax(action_prob) # greedy selection
    s,r,terminated,truncated,info=env.step(a)
    length+=r

    cv.imshow('CartPole animation',cv.cvtColor(env.render(),cv.COLOR_BGR2RGB))
    key=cv.waitKey(100)

    if terminated or truncated:
        print('episode length:',length)
        break

env.close()
if cv.waitKey()==ord('q'):
    cv.destroyAllWindows()
