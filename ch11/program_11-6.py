import gymnasium as gym
import cv2 as cv
from tensorflow.keras.models import load_model

task_name='Hopper-v4'
env=gym.make(task_name,render_mode='rgb_array')

actor=load_model('f11-5.keras')

length=0
s,info=env.reset()
while True:
    mean,_=actor(s.reshape(1,-1))
    s,r,terminated,truncated,info=env.step(mean[0])
    length+=1

    cv.imshow(task_name+' animation',cv.cvtColor(env.render(),cv.COLOR_BGR2RGB))
    key=cv.waitKey(10)

    if terminated or truncated:
        print('episode length:',length)
        break

env.close()
if cv.waitKey()==ord('q'):
    cv.destroyAllWindows()
