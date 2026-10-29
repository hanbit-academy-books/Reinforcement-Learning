import numpy as np

def simulate_pull_bandit(handle,visit,win):
    visit[handle]+=1
    win[handle]+=1 if np.random.random()<arm_prob[handle] else 0

def exploration(): # pure exploration
    visit=np.ones(n_arms) # number of pullings
    win=np.ones(n_arms) # number of winnings
    for t in range(n_trial):
        h=np.random.randint(0,n_arms)
        simulate_pull_bandit(h,visit,win)
    return visit,win

def epsilon_greedy(eps):
    visit=np.ones(n_arms) # number of pullings
    win=np.ones(n_arms) # number of winnings
    for t in range(n_trial):
        if(np.random.random()<eps): # random selection with eps probability
            h=np.random.randint(0,n_arms)
        else:
            p=np.array([win[i]/visit[i] for i in range(n_arms)])
            h=np.random.choice(np.where(p==np.max(p))[0])
        simulate_pull_bandit(h,visit,win)
    return visit,win

n_arms=6 # number of arms
arm_prob=[0.4,0.2,0.1,0.5,0.3,0.6] # winning rates
n_trial=10000 # total number of pullings

visit,win=exploration()
print('statistics by pure exploration:')
print(' winning rates:', ['%6.4f'% (win[i]/visit[i]) for i in range(n_arms)])
print(' profit($):',['%d'% (2*win[i]-visit[i]) for i in range(n_arms)])
print(' net profit($):',sum([2*win[i]-visit[i] for i in range(n_arms)]))

visit,win=epsilon_greedy(0.1)
print('statistics by epsilon-greedy:')
print(' winning rates:', ['%6.4f'% (win[i]/visit[i]) for i in range(n_arms)])
print(' profit($):',['%d'% (2*win[i]-visit[i]) for i in range(n_arms)])
print(' net profit($):',sum([2*win[i]-visit[i] for i in range(n_arms)]))
