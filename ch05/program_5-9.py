import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

greedy_select=lambda x:np.random.choice(np.argwhere(x==np.max(x)).flatten())

n_bin=21 # number of bins
state_bin=[np.linspace(-2.5,2.5,n_bin),
            np.linspace(-4.0,4.0,n_bin),
            np.linspace(-0.28,0.28,n_bin),
            np.linspace(-3.9,3.9,n_bin)]

def state_discretization(state): # converting real-valued state into bin index
    bin_index=[]
    for i in range(len(state)):
        bin_index.append(np.digitize(state[i],state_bin[i])-1)
    return tuple(bin_index)

def Q_learning(env,gamma=0.99,rho=0.1,eps=1.0,eps_decay=0.9999,eps_min=0.05):
    Q=np.zeros(([n_bin]*env.observation_space.shape[0]+[env.action_space.n])) # q initialization
    epi_length=[] # storing episode lengths
    for i in range(100000): # repeat 100000 episodes
        s,info=env.reset()
        s_=state_discretization(s)
        length=0
        while True:
            if np.random.random()<eps:
                a=np.random.randint(0,env.action_space.n)
            else:
                a=greedy_select(Q[s_])
            s1,r,terminated,truncated,info=env.step(a)
            s1_=state_discretization(s1)
            if terminated:
                Q[s_+(a,)]=Q[s_+(a,)]+rho*(r-Q[s_+(a,)])
            else:
                Q[s_+(a,)]=Q[s_+(a,)]+rho*((r+gamma*np.max(Q[s1_]))-Q[s_+(a,)])
            s_=s1_
            length+=1
            eps=max(eps_min,eps*eps_decay)

            if terminated or truncated:
                epi_length.append(length)
                break

            if np.min(epi_length[-5:])>=env.spec.max_episode_steps: # converged if exceeding max length for five consecutive runs
                break
            if i>0 and i%1000==0: print(i,np.mean(epi_length[-1000:]))

    return Q,epi_length,i

env=gym.make('CartPole-v1',render_mode='rgb_array')
Q,epi_length,converge_speed=Q_learning(env)
print(converge_speed,'-th episode converged')
pickle.dump(Q,open('f5-9.pkl','wb'))
env.close()

plt.figure(figsize=(16,5))
plt.plot(range(1,len(epi_length)+1),epi_length)
smooth=np.convolve(epi_length,10*[0.1],mode='valid')
plt.plot(range(1,len(smooth)+1),smooth)
plt.title('Q-learning scores for CartPole')
plt.ylabel('Score')
plt.xlabel('Match')
plt.grid()
plt.show()
