import gymnasium as gym
import matplotlib.pyplot as plt
import time

def generate_episode(env):
    epi=[]
    s,info=env.reset()

    image=env.render()
    plt.imshow(image)
    plt.show()
    time.sleep(0.1)

    while True:
        a=0 # policy always selecting 0 (left)
        s1,r,terminated,truncated,info=env.step(a)
        epi.append([s,a,r]) # appending [state,action,reward]
        s=s1

        image=env.render()
        plt.imshow(image)
        plt.show()
        time.sleep(0.1)

        if terminated or truncated:
            epi.append([s1,None,None])
            break
    return epi

env=gym.make('CartPole-v1',render_mode='rgb_array')

episode=generate_episode(env)
print('episode generated with policy always selecting left:')
for v in episode:
            print(v)

env.close()
