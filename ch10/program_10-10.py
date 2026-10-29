import gymnasium as gym
import cv2 as cv
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense

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

activation=tf.keras.activations.tanh
initializer=tf.keras.initializers.Orthogonal

task_name='Humanoid-v4'
env=gym.make(task_name,render_mode='rgb_array')
s_dim=env.observation_space.shape[0]
a_dim=env.action_space.shape[0]

actor=ActorNetwork()
actor.load_weights('f10-9.weights.h5')

length=0
s,info=env.reset()
while True:
    mean=actor(s.reshape(1,-1))
    s,r,terminated,truncated,info=env.step(mean[0]) # take mean as action (greedy selection)
    length+=1

    cv.imshow(task_name+' animation',cv.cvtColor(env.render(),cv.COLOR_BGR2RGB))
    key=cv.waitKey(10)

    if terminated or truncated:
        print('episode length:',length)
        break

env.close()
if cv.waitKey()==ord('q'):
    cv.destroyAllWindows()
