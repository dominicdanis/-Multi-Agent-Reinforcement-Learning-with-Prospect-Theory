# %%
import numpy as np
from itertools import combinations
from itertools import permutations 
from itertools import product
from numpy import linalg as LA
from numpy.linalg import matrix_power
from numpy.linalg import inv
import random
import copy 
import timeit
import itertools as it
import matplotlib.pyplot as plt

# %%
#Define the Example parameters here
L = 15 #Size of the square grid (Grid numbering Start from bottom left with '0', ..., 'L-1', then right)
n_agents = 4 # number of agents in simulation

totObs = []
                            # the obstacle is a square region
Obs1_start = 18             # the start grid of the first row of obstacle
Obs1_end = 21               # the end grid of the first row of obstacle 
Obs1 = list(range(Obs1_start,Obs1_end)) # the obstacle region
for o in range(1,2):
    Obs1 = Obs1 + list(range(Obs1_start +L*o,Obs1_end+L*o))
totObs.extend(Obs1)
    
Obs2_start = 25
Obs2_end = 28
Obs2 = list(range(Obs2_start,Obs2_end))
for o in range(1,2):
    Obs2 = Obs2 + list(range(Obs2_start +L*o,Obs2_end+L*o))
totObs.extend(Obs2)
    
Obs3_start = 62
Obs3_end = 67
Obs3 = list(range(Obs3_start,Obs3_end))
for o in range(1,1):
    Obs3 = Obs3 + list(range(Obs3_start +L*o,Obs3_end+L*o))
totObs.extend(Obs3)

Obs4_start = 69
Obs4_end = 74
Obs4 = list(range(Obs4_start,Obs4_end))
for o in range(1,1):
    Obs4 = Obs4 + list(range(Obs4_start +L*o,Obs4_end+L*o))
totObs.extend(Obs4)

Obs5_start = 109
Obs5_end = 114
Obs5 = list(range(Obs5_start,Obs5_end))
for o in range(1,4):
    Obs5 = Obs5 + list(range(Obs5_start +L*o,Obs5_end+L*o))
totObs.extend(Obs5)

Obs6_start = 191
Obs6_end = 193
Obs6 = list(range(Obs6_start,Obs6_end))
for o in range(1,2):            # All obstacles contained in one list
    Obs6 = Obs6 + list(range(Obs6_start +L*o,Obs6_end+L*o))
totObs.extend(Obs6)
    
B = []                          # Define target as all agents at goal state
for i in range(n_agents):
    B.append(L**2-1)

c = 0.9 #Control parameter: With this probability agent will move to the grid position with 1-c probability to adjecent grid positions
action_set= [0, 1, 2, 3] #Left, Down, Right, up
state_set = list(range(0,L**2))

len_of_SS = len(state_set)
len_of_AS = len(action_set)
len_of_OS = 16  # Combinations of binary set [up, down, left, right]

# %%
#Identifying corners and edges in the grid 
Corners = [0, L-1, L**2-L, L**2-1]
Left_Side = list(range(L,(L-1)*L,L))
Bottom = list(range(1,L-1))
Right_Side = list(range(2*L-1,L**2-1,L))
Top = list(range(L**2-L+1,L**2-1))

# %%
#Given current state provide all admissible actions 
def admissible_actions(current_state):
    admissible_action_profile = []
 
    if current_state in Corners:  #Corners
        Corner_ID = Corners.index(current_state)
        if Corner_ID == 0:
            admissible_action_profile = [2,3]
                
        elif Corner_ID == 1:
            admissible_action_profile = [0,3]
                    
        elif Corner_ID == 2:
            admissible_action_profile = [0,1]
                
        elif Corner_ID == 3:
            admissible_action_profile = [1,2]
                
        else:
            print('ERROR')
        
    elif current_state in Left_Side:  #Left
        admissible_action_profile = [1,2,3]
            
    elif current_state in Bottom:  #Bottom
        admissible_action_profile = [0,2,3]
            
    elif current_state in Right_Side:  #Right
        admissible_action_profile = [0,1,3]
            
    elif current_state in Top:  #Top
        admissible_action_profile = [0,1,2]
            
    else: #All actions are admissible 
        admissible_action_profile = [0,1,2,3]
               
   
    return admissible_action_profile
    

# %%
#Calculate the neighbor states given current state
def neighboring_states(current_state):
    
    if current_state in Corners:  #Corners
        Corner_ID = Corners.index(current_state)
        if Corner_ID == 0:
            neighbor_states = [current_state, current_state + L, current_state + 1]
                
        elif Corner_ID == 1:
            neighbor_states = [current_state, current_state + L, current_state - 1]
                    
        elif Corner_ID == 2:
            neighbor_states = [current_state, current_state - L, current_state + 1]
                
        elif Corner_ID == 3:
            neighbor_states = [current_state, current_state - L, current_state - 1]
                
        else:
            print('ERROR')
        
    elif current_state in Left_Side:  #Left
        neighbor_states = [current_state, current_state + L, current_state - L, current_state + 1]
            
    elif current_state in Bottom:  #Bottom
        neighbor_states = [current_state, current_state + L, current_state + 1, current_state - 1]
            
    elif current_state in Right_Side:  #Right
        neighbor_states = [current_state, current_state + L, current_state - L, current_state - 1]
            
    elif current_state in Top:  #Top
        neighbor_states = [current_state, current_state - L, current_state + 1, current_state - 1]
            
    else: 
        neighbor_states = [current_state, current_state + L, current_state - L, current_state + 1, current_state - 1]
               
   
    return neighbor_states

# %%
def adjust_action(current_state, action):
    
    if action in admissible_actions(current_state):
        adjusted_action = action
    else:
        if current_state in Corners:  #Corners
            Corner_ID = Corners.index(current_state)
            if Corner_ID == 0:
                adjusted_action = 2
                
            elif Corner_ID == 1:
                adjusted_action = 3
                    
            elif Corner_ID == 2:
                adjusted_action = 1
                
            elif Corner_ID == 3:
                adjusted_action = 1
                
            else:
                print('ERROR')
        
        elif current_state in Left_Side:
            adjusted_action = 2
        elif current_state in Bottom:
            adjusted_action = 2
        elif current_state in Right_Side:
            adjusted_action = 3
        elif current_state in Top:
            adjusted_action = 1
        else:
            adjusted_action = action
            
    return adjusted_action

# %%
# calculating transition matrix
TP = np.zeros((len_of_AS,len_of_SS,len_of_SS))

# if action aa = 0, i.e., moving towards left
for ss in state_set:
    neighbor_states = neighboring_states(ss)
    N = len(neighbor_states)
    if ss in Corners:
        Corner_ID = Corners.index(ss)
        if Corner_ID == 0:
            for ii in neighbor_states:
                TP[0][ss,ii] = 1/N
                
        elif Corner_ID == 1:
            for ii in neighbor_states:
                if ii == ss - 1:
                    TP[0][ss,ii] = c
                else:
                    TP[0][ss,ii] = (1-c)/(N-1)
                    
        elif Corner_ID == 2:
            for ii in neighbor_states:
                TP[0][ss,ii] = 1/N   
        elif Corner_ID == 3:
            for ii in neighbor_states:
                if ii == ss - 1:
                    TP[0][ss,ii] = c
                else:
                    TP[0][ss,ii] = (1-c)/(N-1)
    elif ss in Left_Side:
        for ii in neighbor_states:
                TP[0][ss,ii] = 1/N
    else:
        for ii in neighbor_states:
            if ii == ss - 1:
                TP[0][ss,ii] = c
            else:
                TP[0][ss,ii] = (1-c)/(N-1)

# if action aa = 1, i.e., moving towards down
for ss in state_set:
    neighbor_states = neighboring_states(ss)
    N = len(neighbor_states)
    if ss in Corners:
        Corner_ID = Corners.index(ss)
        if Corner_ID == 0:
            for ii in neighbor_states:
                TP[1][ss,ii] = 1/N
                
        elif Corner_ID == 1:
            for ii in neighbor_states:
                TP[1][ss,ii] = 1/N
                    
        elif Corner_ID == 2:
            for ii in neighbor_states:
                if ii == ss - L:
                    TP[1][ss,ii] = c
                else:
                    TP[1][ss,ii] = (1-c)/(N-1)
        elif Corner_ID == 3:
            for ii in neighbor_states:
                if ii == ss - L:
                    TP[1][ss,ii] = c
                else:
                    TP[1][ss,ii] = (1-c)/(N-1)
    elif ss in Bottom:
        for ii in neighbor_states:
                TP[1][ss,ii] = 1/N
    else:
        for ii in neighbor_states:
            if ii == ss - L:
                TP[1][ss,ii] = c
            else:
                TP[1][ss,ii] = (1-c)/(N-1)
                
# if action aa = 2, i.e., moving towards right
for ss in state_set:
    neighbor_states = neighboring_states(ss)
    N = len(neighbor_states)
    if ss in Corners:
        Corner_ID = Corners.index(ss)
        if Corner_ID == 0:
            for ii in neighbor_states:
                if ii == ss + 1:
                    TP[2][ss,ii] = c
                else:
                    TP[2][ss,ii] = (1-c)/(N-1)
                
        elif Corner_ID == 1:
            for ii in neighbor_states:
                TP[2][ss,ii] = 1/N
                    
        elif Corner_ID == 2:
            for ii in neighbor_states:
                if ii == ss + 1:
                    TP[2][ss,ii] = c
                else:
                    TP[2][ss,ii] = (1-c)/(N-1)
        elif Corner_ID == 3:
            for ii in neighbor_states:
                TP[2][ss,ii] = 1/N
    elif ss in Right_Side:
        for ii in neighbor_states:
                TP[2][ss,ii] = 1/N
    else:
        for ii in neighbor_states:
            if ii == ss + 1:
                TP[2][ss,ii] = c
            else:
                TP[2][ss,ii] = (1-c)/(N-1)

# if action aa = 3, i.e., moving towards up
for ss in state_set:
    neighbor_states = neighboring_states(ss)
    N = len(neighbor_states)
    if ss in Corners:
        Corner_ID = Corners.index(ss)
        if Corner_ID == 0:
            for ii in neighbor_states:
                if ii == ss + L:
                    TP[3][ss,ii] = c
                else:
                    TP[3][ss,ii] = (1-c)/(N-1)
                
        elif Corner_ID == 1:
            for ii in neighbor_states:
                if ii == ss + L:
                    TP[3][ss,ii] = c
                else:
                    TP[3][ss,ii] = (1-c)/(N-1)
                    
        elif Corner_ID == 2:
            for ii in neighbor_states:
                TP[3][ss,ii] = 1/N
        elif Corner_ID == 3:
            for ii in neighbor_states:
                TP[3][ss,ii] = 1/N
    elif ss in Top:
        for ii in neighbor_states:
                TP[3][ss,ii] = 1/N
    else:
        for ii in neighbor_states:
            if ii == ss + L:
                TP[3][ss,ii] = c
            else:
                TP[3][ss,ii] = (1-c)/(N-1)

for aa in action_set:
    for ss in state_set:
        TP[aa][ss,:] = TP[aa][ss,:]/np.sum(TP[aa][ss,:])

# %%
# Check if any agents have collided with each other
def hasCollided(state):
    agent = state[0]
    others = state[1:]
    collided = False
    for location in others:
        if location == agent and location!=L**2-1:
            collided = True
    return collided

# %%
# define the cost matrix 

r = 20
R = np.full((len_of_AS,len_of_SS,len_of_SS),10)
r_temp = np.ones((len_of_SS,len_of_SS))

for ii in state_set:
    for jj in state_set:
        if jj in totObs:
            r_temp[ii,jj] = r
        if jj in B:
            r_temp[ii,jj] = -10
for aa in action_set:
    R[aa] = np.copy(r_temp)
    
# convert the R matrix to the form of state-action pair
reward = np.zeros((len_of_SS,len_of_AS))
for ii in state_set:
    for aa in action_set:
        reward[ii,aa] = np.dot(TP[aa][ii,:],R[aa][ii,:])

# %%
# Collision Penalty
def collision_cost(global_state):
    penalty = 0
    inTerm = 0
    for x in global_state:
        if x == 24: inTerm = inTerm+1
    if len(global_state) == len(set(global_state)):
        penalty = 0
    elif inTerm < 2:
        penalty = 10
    return penalty

# %%
# Scanning and encoding of the global state

from glob import glob

# Check all one step locations for whether they are occupied returns binary list [occupied for each radar state]
def scan(position, globalState):
    locations = np.zeros(L**2,int)
    for x in globalState: locations[x] = 1
    scan_locations = [position+2*L,position+L-1,position+L,position+L+1,position-2,position-1,position+1,position+2,
                      position-L-1,position-L,position-L+1,position-2*L]
    scanned = []
    for i in scan_locations:
        if i>(L**2-1) or i<0:
            scanned.append(0)
            continue
        else:
            scanned.append(locations[i])
    if position-1 in Left_Side:
        scanned[4] = 0
    elif position in Left_Side:
        scanned[1] = 0
        scanned[4] = 0
        scanned[5] = 0
        scanned[8] = 0
    elif position+1 in Right_Side:
        scanned[7] = 0
    elif position in Right_Side:
        scanned[6] = 0
        scanned[7] = 0
        scanned[3] = 0
        scanned[10] = 0
    return scanned

# Convert the scanned output [occupied for each radar state] to a 0-15 representation of which directions have collision potential
def decode(scanned):
    # directions where a collision is possible [up, down, left, right]
    directions = [0, 0, 0, 0]
    if sum(scanned[0:3]) > 0:
        directions[0] = 1
    if sum(scanned[8:]) > 0:
        directions[1] = 1
    if (scanned[1]+sum(scanned[4:5])+scanned[8]) > 0:
        directions[2] = 1
    if (scanned[3]+sum(scanned[6:7])+scanned[10]) > 0:
        directions[3] = 1
    singleState = 0
    for i in range(len(directions)):
        singleState = singleState + directions[i]*2**i
    return singleState

# Observe the one step region and return the 0-15 representation - wrapper function for scan/decode
def observation(position, globalState):
    scanned = scan(position, globalState)
    return decode(scanned)


# %%
# CPT-Estimation
def CPT_Estimation(s, totA, N_max, gamma, V, sigma, landa, eta_1, eta_2, TP, agent_idx, glb_state):
    X_0 = 100000
    X = np.zeros(N_max)
    neighbors = []
    p = []
    # populate next states and their probabilities
    for agent in range(n_agents): 
        neighbors.append(neighboring_states(glb_state[agent]))
        p.append(np.zeros((len(neighbors[agent]))))
    for agent in range(n_agents):
        for s_prime_index in range(0,len(neighbors[agent])):
            p[agent][s_prime_index] = TP[totA[agent_idx]][glb_state[agent],neighbors[agent][s_prime_index]]
        
    for ii in range(0,N_max):
        s_prime = np.zeros(n_agents, dtype=int)
        for agent in range(n_agents):
            s_prime[agent] = random.choices(neighbors[agent], weights = p[agent], k=1)[0]
            glb_state = s_prime

        obs_prime = observation(s_prime[agent_idx],s_prime)
        X[ii] = reward[s,totA[agent_idx]] + collision_cost(glb_state) + gamma*V[s_prime[agent_idx]][obs_prime] + random.gauss(0,1)
        if X[ii] < X_0:
            s_star = s_prime[agent_idx]
            X_0 = X[ii]
    
    rho_plus = 0
    rho_minus = 0
    X_sort = np.sort(X, axis = None)
    
    for ii in range(0,N_max):
        z_1 = (N_max + ii - 1)/N_max
        z_2 = (N_max - ii)/N_max
        z_3 = ii/N_max
        z_4 = (ii-1)/N_max
        rho_plus = rho_plus + abs(max(0,X_sort[ii]))**sigma * (z_1**eta_1/(z_1**eta_1 + (1-z_1)**eta_1)**(1/eta_1)-z_2**eta_1/(z_2**eta_1 + (1-z_2)**eta_1)**(1/eta_1))
        rho_minus = rho_minus + abs(min(0,X_sort[ii]))**sigma * (z_3**eta_2/(z_3**eta_2 + (1-z_3)**eta_2)**(1/eta_2)-z_4**eta_2/(z_4**eta_2 + (1-z_4)**eta_2)**(1/eta_2))
    rho = rho_plus - rho_minus

    return rho_plus, s_star    

# %%
# given a set of learned policies test them out, return total cost and total collisions - for a set of policies (multi-agent)
def multi_simulate_runs(policies, init_state, TP, R):
    current_action = np.zeros(len(policies),dtype=int)
    current_state = init_state
    next_state = np.zeros(len(policies),dtype=int).tolist()

    if len(state_set) > 1000:
        time_horizon = 40*len(state_set)
    else:
        time_horizon = 100*len(state_set)
    counter = 0
    collision = 0
    cost = 0
    inTerminal = False

    while not inTerminal and counter <= time_horizon:
        for i in range(len(policies)):
            if current_state[i]!=L**2-1:
                obs = observation(current_state[i],current_state)
                current_action[i] = policies[i][obs][current_state[i]]
                current_action[i] = adjust_action(current_state[i], current_action[i])
                next_state[i] = random.choices(state_set, weights = TP[current_action[i]][current_state[i],:], k = 1)[0]
                cost = cost + gamma**counter*(R[current_action[i]][current_state[i]][next_state[i]]+collision_cost(next_state))
                current_state[i] = next_state[i]
        if np.any(next_state in totObs) or hasCollided(next_state):
            collision = collision + 1
        counter = counter + 1
        if current_state == B:
            inTerminal = True
    return collision, cost

# %%
# Helper function to flip a weighed coin - there may be a library function somehwere for this
def flip(p):
    return True if random.random() > p else False


print('Env set up')

# %%
# CPT Weight Sharing Strategy

Q = np.zeros((n_agents,len_of_SS, len_of_OS, len_of_AS))
V = np.amin(Q[:][:][:][:], axis = 3)
expertness = np.zeros(n_agents)
T = np.full((n_agents),1)
T_max = 1000
sharingIterations = 10
gamma = 0.9
N_max = 100
sigma = 0.88
landa = 0.25
eta_1 = 0.61
eta_2 = 0.69
lr = 0.1
expertness = np.zeros(n_agents)
epsilon = np.full((n_agents),1,float)

globalState = [random.randint(0,len_of_SS-1) for i in range(n_agents)]
while np.any(T < T_max):
    if np.any(T<T_max-sharingIterations):      
        s_next = []
        totalActions = []
        observed = np.zeros(n_agents,int)
        # choose actions for each agent
        for agent in globalState:
            i = globalState.index(agent)
            actions = admissible_actions(agent)
            observed[i] = observation(agent,globalState)
            greedy = flip(epsilon[i])                              # use epsilon greedy
            if greedy:
                a = np.argmin(Q[i,agent,observed[i], :])
            else:
                a = random.choice(actions)
            adj_action = adjust_action(agent,a)
            totalActions.append(adj_action)

        # get rewards for each agents actions, update Q values and move agents
        for agent in globalState:
            i = globalState.index(agent)
            rho, s_star = CPT_Estimation(agent, totalActions, N_max, gamma, V[i], sigma, landa, eta_1, eta_2, TP, i, globalState)
            delta = rho - Q[i,agent,observed[i],a] # CPT implemenation
            expertness[i] = expertness[i] + delta
            Q[i,agent,observed[i],a] = (1-lr) * Q[i,agent,observed[i],a] + lr*delta    # Q value update

            # testing this out
            if agent == B[0]:
                epsilon[i] = epsilon[i]*0.998
                s_star = random.randint(0,len_of_SS-1)
                T[i] = T[i] + 1
    
            s_next.append(s_star)
        globalState = s_next
    else:
        for i in range(n_agents):
            weights = expertness/sum(expertness)
            Qnew = np.zeros((len_of_SS, len_of_OS, len_of_AS))
            for n in range(n_agents):
                Q[i,:,:,:] = Qnew + weights[n]*Q[n,:,:,:]
            T = T +1

# pi_proposed is [agents][observation][state]
pi_proposed = np.zeros((n_agents,len_of_OS,len_of_SS),dtype=int)

for i in range(n_agents):                                       # extract policy from q values
    for o in range(len_of_OS):
        pi_proposed[i][o][:] = np.argmin(Q[i,:,o,:],axis = 1)
        for s in range(len_of_SS):
           pi_proposed[i][o][s] = adjust_action(s,pi_proposed[i][o][s])
        

simulation_number = 100
collision_count_CPTWSS = np.zeros(simulation_number)
cost_count_CPTWSS = np.zeros(simulation_number)

print('Simulation Start')

for z in range(0,simulation_number):
    collision_count_CPTWSS[z], cost_count_CPTWSS[z] = multi_simulate_runs(pi_proposed, list(range(n_agents)), TP, R )


# save the number of collisions of 100 trajectories of CPT-WSS
fname = 'collision_count_CPTWSS.txt'
np.savetxt(fname, collision_count_CPTWSS)

# save the cost of 100 trajectories of CPT-WSS
fname = 'cost_count_CPTWSS.txt'
np.savetxt(fname, cost_count_CPTWSS)
