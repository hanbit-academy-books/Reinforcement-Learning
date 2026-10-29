import gymnasium as gym
import tensorflow as tf
import cv2 as cv

env=gym.make('Hopper-v4',render_mode='rgb_array')

policy_bc_vanilla=tf.keras.models.load_model('f12-1.keras')

length=0
s,info=env.reset()
while True:
    action=policy_bc_vanilla(s.reshape(1,-1))[0]
    s1,r,terminated,truncated,info=env.step(action)
    s=s1
    length+=1

    cv.imshow('Hopper animation',cv.cvtColor(env.render(),cv.COLOR_BGR2RGB))
    key=cv.waitKey(10)

    if terminated or truncated:
        print('episode length:',length)
        break

env.close()
if cv.waitKey()==ord('q'):
    cv.destroyAllWindows()
