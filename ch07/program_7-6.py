import gymnasium as gym
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import cv2 as cv
import ale_py

consecutive=4 # represent state with 4 consecutive frames
rows=84
cols=84
actions=[0,2,3] # 0(stop), 2(up), 3(down)

def preprocess(img):
    return cv.resize(cv.cvtColor(img[30:-12,5:-4,:],cv.COLOR_BGR2GRAY),dsize=(rows,cols)).reshape(rows,cols,1)

def play_match(env):
    obs,info=env.reset()
    imem=deque(maxlen=consecutive) # represent state with 4 consecutive frames
    for j in range(consecutive): imem.append(preprocess(obs))
    s=np.transpose(np.squeeze(np.asarray(imem)),(1,2,0))
    while True:
        y_hat=model(s.reshape(1,rows,cols,consecutive))[0]
        a=np.argmax(y_hat)
        obs,reward,terminated,truncated,info=env.step(actions[a])
        imem.append(preprocess(obs))
        s=np.transpose(np.squeeze(np.asarray(imem)),(1,2,0))

        cv.imshow('animation',cv.cvtColor(env.render(),cv.COLOR_BGR2RGB))
        key=cv.waitKey(10)
        if key==ord('q'): break

        if terminated or truncated:
            break

model=load_model('f7-5.keras')
env=gym.make('ALE/Pong-v5',render_mode='rgb_array')
play_match(env)
env.close()
if cv.waitKey()==ord('q'):
    cv.destroyAllWindows()
