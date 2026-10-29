import numpy as np

def advantage_MC(rewards,values,gamma): # Monte Carlo advantage
    T=len(rewards)
    returns=np.zeros(T)
    future_return=0
    for t in reversed(range(T)):
        future_return=rewards[t]+gamma*future_return
        returns[t]=future_return
    advantages=returns-np.array(values)
    return advantages

def advantage_n_step(rewards,values,gamma,n): # n-step advantage
    T=len(rewards)
    advantages=np.zeros(T)
    for t in range(T):
        n_step_return=0
        for k in range(n):
            if t+k<T:
                n_step_return+=(gamma**k)*rewards[t+k]
        if t+n<T:
            n_step_return+=(gamma**n)*values[t+n]
        advantages[t]=n_step_return-values[t]
    return advantages

def advantage_GAE(rewards,values,gamma,lamda): # GAE advantage
    T=len(rewards)
    deltas=np.zeros(T)
    for t in range(T-1):
        deltas[t]=rewards[t]+gamma*values[t+1]-values[t]
    deltas[-1]=rewards[-1]-values[-1] # termination state

    advantages=np.zeros(T)
    advantage=0
    for t in reversed(range(T)):
        advantage=deltas[t]+gamma*lamda*advantage
        advantages[t]=advantage
    return advantages

rewards=[1,1,1,1,1,1,1]
values=[0.4,0.7,1.1,0.9,0.5,1.2,0.8]

adv_MC=advantage_MC(rewards,values,0.99)
adv_n_step=advantage_n_step(rewards,values,0.99,3)
adv_GAE=advantage_GAE(rewards,values,0.99,0.95)
print('Monte Carlo advantage',adv_MC)
print('n-step advantage ',adv_n_step)
print('GAE advantage ',adv_GAE)
