'''
This file contains the Deep Q Learning Neural Net (DQN) class, as well as a class containing methods used by the DQN model.
'''
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import math
import networkx as nx
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.animation as animation



R_EARTH = 3958.8                # Radius of Earth in miles

OUTPUT_SIZE = 1                 # Length of NN output
NUM_FEATURES = 6                # Length of NN input (# of features)



'''
This model stands as the basis of the most developed pathing learners.
'''
class DQN(nn.Module):
    def __init__(self, input_size=NUM_FEATURES, hidden_size=2*NUM_FEATURES):
        super(DQN, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.LeakyReLU(0.2)
        self.fc2 = torch.nn.Linear(self.hidden_size, OUTPUT_SIZE)
        self.sigmoid = torch.nn.Sigmoid()
            
        self.iter = 0
            
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


'''
The following class stores all the necessary methods used for both training and evaluating a DQN model on a given environment (Map).
'''
class METHODS:
    '''
    The initializer method creates a reference to the environment, model, as well as other hyperparameters used later.
    '''
    def __init__(self, Map, model, clusters, EPSILON_START=5, EPSILON_CONST=0.25, EPSILON_STATIC=False, lr=0.1, BATCH_SIZE=7, START_NODE=0):
        self.Map = Map                                                      # NetworkX DiGraph of the region
        self.model = model                                                  # DQN instance; model to be trained/evaluated
        self.clusters = clusters                                            # List of clusters [C[0]: coordinates, C[1]: size of cluster]   <- C: cluster

        self.EPSILON_START  = EPSILON_START                                 # Initial epsilon value for dynamic epsilon greedy
        self.EPSILON_CONST  = EPSILON_CONST                                 # Constant epsilon value for static epsilon greedy
        self.EPSILON_STATIC = EPSILON_STATIC                                # Boolean value that determines type of epsilon value
        
        self.lr = lr                                                        # Learning rate of NN
        
        self.FULL_SIZE = len(self.clusters)                                 # Full size of cluster dataset
        self.BATCH_SIZE = min(BATCH_SIZE, 10, self.FULL_SIZE-1)             # Batch size when training learner [MAX VALUE: 10 / number of clusters - start node]

        self.START_NODE = START_NODE                                        # Index of starting cluster

        self.DISTANCE_TABLE = np.array([[self.total_distance(i, j) for i in range(len(self.clusters))] for j in range(len(self.clusters))])
        self.memo = []                                                      # Memoization for iterator: used to prune bad branches and speed up process
    


    # Takes in indices of clusters and returns the total distance
    def total_distance(self, i, j):
        distance = 0
        path = nx.shortest_path(self.Map, self.clusters[i][0], self.clusters[j][0])

        for k in range(1, len(path)): # Identifies the edge, stores the distance and adds it to the total distance
            distance += self.Map.edges[path[k-1], path[k]]['Distance']
        return distance



    '''
    The following method returns the normal dot product (also the cos of the angle) between two edges.
    If given a z coordinate, it will measure the angle of deviation the pather takes from where it originally was headed.
    If given a CORE coordinate, it will measure the angle between the next action and the CORE.
    Both values shouldn't be called simultaneously, however one value must be given.
    '''
    def norm_dot_product(self, x, y, z=None, CORE=None): # x, y are start / end coordinates of a path (respectively)
        # z is called whenever an angle is formed between two edges; CORE is called whenever comparing the angle to the current average cluster point
    
        norm = lambda x, y: np.sqrt((y[0]-x[0])**2 + (y[1]-x[1])**2)
    
        if z==None:
            return ( (y[0]-x[0])*(CORE[0]-x[0]) + (y[1]-x[1])*(CORE[1]-x[1]) ) / ( norm(x, CORE)*norm(x, y) )
        return ( (y[0]-x[0])*(z[0]-y[0]) + (y[1]-x[1])*(z[1]-y[1]) ) / ( norm(x, y)*norm(y, z) )



    # Generates a random batch of actions within range of FULL_SIZE and of size BATCH_SIZE
    def get_batch(self):
        temp = [i for i in range(self.FULL_SIZE)]
        temp.remove(self.START_NODE) # we don't include the start node
        x = []
    
        for _ in range(self.BATCH_SIZE):
            x.append(temp.pop(random.randint(0, len(temp)-1)))
        return x



    # Finds the optimal path given a specific state
    def iterator(self, state, reset=False):
        if reset:
            self.memo.clear()
        
        if state[2]==[]: # End state
            return [state[3], state[1]]
    
        for s in self.memo: # Runs through memory
            if s[0]==state[0] and set(s[1])==set(state[1]): # if they have same current position and similar past experiences...
                if state[3] >= s[3]: # stop if greater travel distance
                    return [1e99, []]
                self.memo.remove(s) # replace memory value if better
                break
            
        self.memo.append(state) # add state to memory
    
        return min( self.iterator( (state[2][i], state[1]+[state[2][i]], state[2][:i]+state[2][i+1:], state[3]+self.DISTANCE_TABLE[state[0], state[2][i]])) for i in range(len(state[2])) )



    '''
    This method takes a state-action pair, and returns a tensor with its feature components, which include:
    1. Distance between current node and [action] node
    2. Cosine of angle of deviation from previous action to [action] node (more info at norm_dot_product())
    3. Cosine of angle between [action] and average of remaining coordinates
    4. Distance to the average of remaining coordinates
    5. Progress finished (# of nodes finished / # of total nodes)
    6. Ranked percentage of distance to [action] compared to others (0: closest, 1: furthest)
    '''
    def extract_features(self, state, action): #action: node we're considering traveling to (index)
        # Measure existing "node-core" (average point of remaining clusters)
        clusters_remaining_x = [self.clusters[i][0][0] for i in state[2]+[action]]
        clusters_remaining_y = [self.clusters[i][0][1] for i in state[2]+[action]]
        CORE = sum(clusters_remaining_x)/(len(state[2])+1), sum(clusters_remaining_y)/(len(state[2])+1)
    
        distance = self.DISTANCE_TABLE[state[0], action] / R_EARTH
    
        angle_from_prev = 0
        if len(state[1])>1:
            angle_from_prev = self.norm_dot_product(self.clusters[state[1][-2]][0], self.clusters[state[0]][0], self.clusters[action][0])
        
        angle_to_core = self.norm_dot_product(self.clusters[state[0]][0], self.clusters[action][0], CORE=CORE)
        norm = lambda x, y: np.sqrt((y[0]-x[0])**2 + (y[1]-x[1])**2)
        distance_to_core = norm(self.clusters[state[0]][0], CORE)
    
        percent_visited = len(state[1]) / (len(state[1])+len(state[2]))
    
        choices = [(self.DISTANCE_TABLE[state[0], a], a) for a in state[2]+[action]]
        choices.sort()
        percent_closest = [i+1 for i in range(len(state[2])+1) if choices[i][1]==action][0] / (len(state[2])+1)
    
        x = [ distance, angle_from_prev, angle_to_core, distance_to_core, percent_visited, percent_closest ]
    
        return torch.FloatTensor(x)



    '''
    This method is epsilon-greedy:
    return random action                                                    if within specified range
    return action with highest Q value                                      otherwise
    '''
    def select_action(self, state, opt=False): # opt: no exploration
        self.model.iter += 1
    
        if self.EPSILON_STATIC: # If using a constant epsilon value, run the following
            if random.random() > self.EPSILON_CONST or opt:
                output = torch.FloatTensor([self.model.forward(self.extract_features(state, action)) for action in state[2]])
                return state[2].pop( torch.argmax(output) )
            else:
                return state[2].pop(random.randint(0, len(state[2])-1))
    
        # Other use the dynamic epsilon value
        if random.random() > self.EPSILON_START/math.sqrt(self.model.iter) or opt:
            output = torch.FloatTensor([self.model.forward(self.extract_features(state, action)) for action in state[2]])
            return state[2].pop( torch.argmax(output) )
        else:
            return state[2].pop(random.randint(0, len(state[2])-1))



    # Updates model given a certain outcome
    def incorporate_feedback(self, state, action, reward):
        x = self.extract_features(state, action)
        Q = self.model.forward(x)
    
        criterion = nn.MSELoss()
    
        loss = criterion(Q, reward*torch.ones(OUTPUT_SIZE)) # Loss = (Q-I[x])^2
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return



    '''
    Trains method over a set number of episodes. [defaults to 500]
    Verbose determines the printed output of the training:
    0: no output
    1: Episode #
    2: Episode # and Learner vs. Optimal paths
    '''
    def train(self, num_episodes = 500, verbose=2):
        SCORES = [] # denotes total rewards of model for each episode

        for i in range(num_episodes):
            if verbose>0:
                print("Episode #", i)
            SCORES.append(0)
    
            # First restart the 'game'
            actions = self.get_batch() # Actions are the indices of clusters
            state = (self.START_NODE, [self.START_NODE], actions, 0)
            PI_opt = self.iterator(state, True)[1] # The optimal policy for the batch
            if verbose>1:
                print('Optimal Path:', PI_opt)
    
            while state!=None:
                action = self.select_action(state)
                optimal_action = PI_opt[len(state[1])]
        
                D = self.DISTANCE_TABLE[state[0], action]
        
                newState = (action, state[1]+[action], state[2], state[3]+D)
        
                if action!=PI_opt[len(state[1])]: #update the optimal policy if a different action was taken
                    PI_opt = self.iterator(newState, True)[1]
        
                reward = 2*int(action==optimal_action)-1 # reward is 1 if learner replicates PI_opt, -1 otherwise
        
                if len(newState[2])==0: # If no new possible actions, set newState to None
                    if verbose>1:
                        print('Learner Path:', newState[1])
                    newState = None
        
                self.incorporate_feedback(state, action, reward)
                SCORES[-1] += reward
        
                state = newState
        return SCORES



    '''
    This method will iterate the trained model over the full environment and return...
    1: Total distance travelled
    2: Actions taken

    Verbose=1: will output all intermediate distances when chosen
    '''
    def eval(self, verbose=1):
        actions = [i for i in range(self.FULL_SIZE)] # Now we're working with the full set of clusters
        actions.remove(self.START_NODE)
        state = (self.START_NODE, [self.START_NODE], actions, 0)

        traveled_distance = 0

        if len(state[2])==0:
            return traveled_distance, actions
        
        actions = [self.START_NODE] # We will now re-utilize actions as a memory of the actions taken in order
        while state!=None:
            action = self.select_action(state, opt=True)
            actions.append(action)
        
            D = self.DISTANCE_TABLE[state[0], action]
            traveled_distance += D
            if verbose==1:
                print(D) # This print statement allows us to monitor the movements and see notice any unnecessarily long paths
            state = (action, state[1]+[action], state[2], state[3]+D)
            if len(state[2])==0:
                state = None
        
        return traveled_distance, actions



    '''
    This method will run an evaluation, then save the actions as an animation
    regionPath: path to shp file containing road background
    '''
    def getAnimation(self, regionPath, animationName, RRC=True, line_length=50, frameSkip=1, frameLimit=1e99, fps=2500):
        _, actions = self.eval(verbose=0)
        
        directions_x = [[] for _ in range(len(actions)-1)]
        directions_y = [[] for _ in range(len(actions)-1)]

        for i in range(1, len(actions)):
            p1, p2 = self.clusters[actions[i-1]][0], self.clusters[actions[i]][0]
            temp = nx.shortest_path(self.Map, p1, p2, weight='Distance')
            for e in temp:
                directions_x[i-1].append(e[0])
                directions_y[i-1].append(e[1])
        
        # Basemap will be the background (road map) of the animation
        basemap = gpd.read_file(regionPath)

        fig, ax = plt.subplots(figsize=(15,15))
        xs = []
        ys = []
        line, = ax.plot(xs, ys, color='r', linewidth=3)

        if RRC: # If the data's not from RRC, then we can't use DISP_CODE
            basemap.plot(column='DISP_CODE', ax=ax, figsize=(15,15), zorder=0)
        else:
            basemap.plot(ax=ax, figsize=(15,15), zorder=0)

        for C in self.clusters:
            plt.scatter(C[0][0], C[0][1], s=25*C[1])

        FULL_DIRECTIONS_X, FULL_DIRECTIONS_Y = [x for X in directions_x for x in X ], [y for Y in directions_y for y in Y ]

        # This function is called periodically from FuncAnimation
        def animate(i, xs=xs, ys=ys):
            # Add x and y to lists
            xs.append(FULL_DIRECTIONS_X[i])
            ys.append(FULL_DIRECTIONS_Y[i])

            # Limit x and y lists
            xs = xs[-line_length:]
            ys = ys[-line_length:]

            line.set_data(np.array(xs), np.array(ys))
    
            return line,

        # Set up plot to call animate() function periodically
        ani = animation.FuncAnimation(fig, animate, frames=list(range(0, min(len(FULL_DIRECTIONS_X), frameLimit), frameSkip)) )
        ani.save('Gifs\\'+animationName+'.gif', fps=fps)
