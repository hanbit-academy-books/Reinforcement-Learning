import numpy as np
import scipy

def discount_cumulation(x,discount):
    return scipy.signal.lfilter([1],[1,-discount],x[::-1],axis=0)[::-1]

def advantage_MC(rewards,values,gamma): # Monte Carlo advantage
    return discount_cumulation(rewards,gamma)-np.array(values)

def advantage_GAE(rewards,values,gamma,lamda): # GAE advantage
    rewards,values,deltas=np.array(rewards),np.array(values),np.zeros(len(rewards))
    deltas[:-1]=rewards[:-1]+gamma*values[1:]-values[:-1]
    deltas[-1]=rewards[-1]-values[-1] # termination state
    return discount_cumulation(deltas,gamma*lamda)

rewards=[1,1,1,1,1,1,1]
values=[0.4,0.7,1.1,0.9,0.5,1.2,0.8]

adv_MC=advantage_MC(rewards,values,0.99)
adv_GAE=advantage_GAE(rewards,values,0.99,0.95)
print('Monte Carlo advantage',adv_MC)
print('GAE advantage',adv_GAE)
