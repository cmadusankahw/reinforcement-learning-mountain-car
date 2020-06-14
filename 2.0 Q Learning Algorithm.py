import numpy as np

R=np.matrix([[-1,-1,-1,-1,0,-1],
             [-1,-1,-1,0,-1,100],
             [-1,-1,-1,0,-1,-1],
             [-1,0,0,-1,0,-1],
             [0,-1,-1,0,-1,100],
             [-1,0,-1,-1,0,100]])

gamma=0.8
Q=np.zeros([6,6])

def find_available_actions(current_state):

    current_row=R[current_state,:]
    available_actions=np.where(current_row>=0)[1]
    return available_actions

def find_next_action(av_actions):

    return int(np.random.choice(av_actions,1))

def update_Q(current_state,nx_action,gamma):

    av_actions=find_available_actions(nx_action)
    max_value=max(Q[nx_action,av_actions])
    Q[current_state,nx_action]=R[current_state,nx_action]+gamma*max_value

for i in range(1000): 

    current_state=np.random.randint(0,6)
    av_actions=find_available_actions(current_state)
    nx_action=find_next_action(av_actions)
    update_Q(current_state,nx_action,gamma)

Q=Q/np.max(Q)*100
Q=np.round(Q,2)

print('====TRAINED Q MATRIX====')
print(Q)
