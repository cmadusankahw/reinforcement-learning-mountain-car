import gym
import time as t

env=gym.make('MountainCar-v0')

env.reset()

def get_reduced_state(state):

    reduce_state=(state-env.observation_space.low)/reduced_states_win
    return tuple(reduce_state.astype(np.int))

alpha=0.1                   #alpha
gamma=0.95                  #gamma
episodes=25000

status=False

reduced_states=[20,20]
reduced_states_win=(env.observation_space.high-env.observation_space.low)/reduced_states

import numpy as np

QM=np.random.uniform(low=-2,high=0,size=(reduced_states+[env.action_space.n]))
#declaring the Q-Matrix with random values (-2,0)

print(QM.shape)

reduced_state=get_reduced_state(env.reset())
#setting the initial state, to one of the reduced 20 states


for episode in range(episodes):

    while(status==False):

        action=np.argmax(QM[reduced_state]) #argmax returns the index of the max val
        new_state,reward,status,_ = env.step(action)

        new_reduced_state=get_reduced_state(new_state)
        
        env.render()

        if(status==False):

            currentQ = QM[reduced_state+(action,)]
            maxQNS = np.max(QM[new_reduced_state])
            
            QM[reduced_state+(action,)]=(1-alpha)*currentQ+ alpha *(reward+ gamma*maxQNS)

        else: #if the goal is reached

            QM[reduced_state+(action,)]=0
            reduce_state=new_reduced_state
            action=np.argmax(QM[reduced_state]) #argmax returns the index of the max val
            new_state,reward,status,_ = env.step(action)
            env.render()
            
        reduce_state=new_reduced_state

env.close()
