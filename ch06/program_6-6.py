import gymnasium as gym
import numpy as np
from tensorflow.keras.models import load_model

env=gym.make('Blackjack-v1', render_mode='rgb_array')
s_dim=len(env.observation_space) # number of state variables

q=load_model('f6-4.keras') # load the neural network

print(3*' '+'state'+5*' '+'stick'+4*' '+'hit'+8*' '+'state'+5*' '+'stick'+4*' '+'hit')
dealer_card=8
for player in range(1,22):
    not_usable_ace=(player,dealer_card,0)
    usable_ace=(player,dealer_card,1)
    Q=q.predict(np.array(not_usable_ace).reshape([1,s_dim]),verbose=0)[0]
    q0_stick=Q[0]
    q0_hit=Q[1]
    Q=q.predict(np.array(usable_ace).reshape([1,s_dim]),verbose=0)[0]
    q1_stick=Q[0]
    q1_hit=Q[1]
    print(not_usable_ace,'%7.4f %7.4f'%(q0_stick,q0_hit),' ',end='')
    print(usable_ace,'%7.4f %7.4f'%(q1_stick,q1_hit))
