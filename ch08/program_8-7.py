import gymnasium as gym
from tensorflow.keras.models import load_model
import cv2 as cv
import tensorflow as tf
@tf.keras.utils.register_keras_serializable()

def log_1(x): return tf.math.log(1+tf.math.exp(x))

model=load_model('f8-6.keras')

env=gym.make('InvertedPendulum-v4',render_mode='rgb_array')
s_dim=env.observation_space.shape[0] # state space

length=0
s,info=env.reset()
while True:
    mean,std=model(s.reshape([1,s_dim]))
    a=mean[0] # taking mean (greedy selection)
    s,r,terminated,truncated,info=env.step(a)
    length+=r

    cv.imshow('InvertedPendulum animation',cv.cvtColor(env.render(),cv.COLOR_BGR2RGB))
    key=cv.waitKey(100)

    if terminated or truncated:
        print('episode length:',length)
        break

env.close()
if cv.waitKey()==ord('q'):
    cv.destroyAllWindows()
