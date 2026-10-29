import numpy as np
import matplotlib.pyplot as plt

def simulate_pull_bandit(handle,visit,win):
    visit[handle]+=1
    win[handle]+=1 if np.random.random()<arm_prob[handle] else 0

def exploration_and_exploitation(rate):
    visit=np.ones(n_arms) # number of pullings
    win=np.ones(n_arms) # number of winnings
    for t in range(int(n_trial*rate)): # identifying the best bandit using the specified rate
        h=np.random.randint(0,n_arms)
        simulate_pull_bandit(h,visit,win)
    p=np.array([win[i]/visit[i] for i in range(n_arms)])
    h=np.random.choice(np.where(p==np.max(p))[0])
    for t in range(n_trial-int(n_trial*rate)):
        simulate_pull_bandit(h,visit,win)
    return visit,win

def epsilon_greedy(eps):
    visit=np.ones(n_arms) # number of pullings
    win=np.zeros(n_arms) # number of winnings
    for t in range(n_trial):
        if(np.random.random()<eps): # random selection with eps probability
            h=np.random.randint(0,n_arms)
        else:
            p=np.array([win[i]/visit[i] for i in range(n_arms)])
            h=np.random.choice(np.where(p==np.max(p))[0])
        simulate_pull_bandit(h,visit,win)
    return visit,win

def UCB(c):
    visit=np.ones(n_arms) # number of pullings
    win=np.ones(n_arms) # number of winnings
    for t in range(n_trial):
        p=np.array([win[i]/visit[i]+c*np.sqrt(np.log(t+1)/visit[i]) for i in range(n_arms)]) # Eq (4.1)
        h=np.random.choice(np.where(p==np.max(p))[0])
        simulate_pull_bandit(h,visit,win)
    return visit,win

def box_plot_analysis(algorithm,hyper_param,alg_name):
    net_profit=[]
    for param in hyper_param:
        net=[]
        for k in range(n_run): # multiple runs to draw box plot
            visit,win=algorithm(param)
            net.append(sum([2*win[i]-visit[i] for i in range(n_arms)]))
        net_profit.append((net))
    print(alg_name+' performance:',np.mean(net_profit,axis=1))
    plt.boxplot(net_profit)
    plt.title(alg_name+' performance')
    plt.xlabel('hyper-parameters')
    plt.ylabel('net profits')
    plt.grid()
    plt.xticks(np.arange(1,len(hyper_param)+1),hyper_param)
    plt.show()

n_arms=200 # number of arms
arm_prob=[0.4+(np.random.random()-0.5)/2 for i in range(n_arms)] # winning rates
n_trial=10000 # total number of pullings
n_run=50 # multiple runs for statistical confidence

box_plot_analysis(exploration_and_exploitation,[0.0,0.1,0.25,0.5,0.75,0.99,1.0],'exploration_and_exploitation')
box_plot_analysis(epsilon_greedy,[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],'eps-greedy')
box_plot_analysis(UCB,[0.0,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0],'UCB')
