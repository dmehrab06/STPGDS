import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

EPS = 0.00001
MAX_SCALE = 10 # space will be within (0,0) -- (10,10)
Q = 23.3 #pore change kora jaite pare
v = 0.8 #pore change hoite pare

MAX_TIME = 10

class Region:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def dis(self,r,dis_method='euclidean'):
        if dis_method=='manhattan':
            return abs(self.x-r.x)+abs(self.y-r.y)
        else:
            return ((self.x-r.x)**2+(self.y-r.y)**2)**0.5

class TimeInterval:
    def __init__(self,b,e):
        self.b = b
        self.e = e
        if(self.b>self.e):
            temp = self.b
            self.b = self.e
            self.e = temp
    def time_intersects(self,time):
        if self.b<=time and time<=self.e:
            return True
        return False

def calc_intention(fear):
    return 1.0/(1.0+Q*math.exp(-v*fear))

class Agent:
    def __init__(self, idx, r, beta):
        self.id = idx
        self.region = r
        self.beta = beta #(flee bias beta between [0,1])
        self.fear = 0 #(>0)
        self.intention = calc_intention(self.fear)
        self.migration = 0
    
        
    def update_fear(self,risk,theta):
        if self.migration!=1:
            self.fear = self.fear*theta + risk #(theta - discount [0,1])
            self.intention = calc_intention(self.fear)
            self.migration = 0
        else:
            self.fear = 1000000
            self.intention = 1
            self.migration = 1
    
    def reset_agent(self):
        self.fear = 0
        self.intention = calc_intention(self.fear)
        self.migration = self.intention

    def update_migration_from_peer(self,peer_effect,THRESH):
        if peer_effect>THRESH or self.migration==1:
            self.migration = 1
        else:
            self.migration = 0

class Event:
    def __init__(self, idx, r, w, tint):
        self.event_id = idx
        self.region = r
        self.weight = w
        self.interval = tint
    def happened(self,time):
        if (self.interval).time_intersects(time):
            return True
        return False
    def __str__(self):
        all_info = ('event id:',str(self.event_id),'weight:',str(self.weight),'loc:',str(self.region.x),',',str(self.region.y),'time interval: from',str(self.interval.b),'to',str(self.interval.e))
        return ' '.join(all_info)

def riskfunc(a,E,delta):
    tot = 0.0
    for event in E:
        dis = ((a.region).dis(event.region,dis_method='manhattan'))**delta
        dis = dis+EPS
        cur_fear = (a.beta)*(event.weight)/dis
        tot = tot+cur_fear
    return tot
            

def createplot(Agents,Events,ax,Zoom_Factor=2,Agent_Size=5):
    OFFSET = 0.1
    for a in Agents:
        ax.plot(a.region.x,a.region.y,marker='o',color='blue',markersize=Agent_Size)
    for e in Events:
        ax.plot(e.region.x,e.region.y,marker='X',color='red',markersize=int(e.weight)*Zoom_Factor)
        #ax.annotate(str(e.event_id), xy =(e.region.x,e.region.y), 
        #     xytext =(e.region.x+OFFSET, e.region.y+OFFSET))
    ax.set_xlim([0,MAX_SCALE])
    ax.set_ylim([0,MAX_SCALE])

def create_random_events(NUM_EVENTS):
    #NUM_EVENTS = 10
    MAX_WEIGHT = 5

    VERY_BIG_EVENT_WEIGHT = 10
    big_event = Event(NUM_EVENTS,Region(5,5),VERY_BIG_EVENT_WEIGHT,TimeInterval(0,2))

    ## create events:
    RANDOM_EVENTS = [big_event]
    for i in range(0,NUM_EVENTS):
        cur_id = i
        rand_loc = np.random.rand(1,2).flatten()
        r = Region(rand_loc[0]*MAX_SCALE,rand_loc[1]*MAX_SCALE)
        w = np.random.randint(0,MAX_WEIGHT)
        t1 = np.random.randint(0,MAX_TIME)
        t2 = np.random.randint(0,MAX_TIME)
        tint = TimeInterval(t1,t2)
        RANDOM_EVENTS.append(Event(cur_id,r,w,tint))
    return RANDOM_EVENTS

def create_same_events_circle(NUM_EVENTS,CIRCLE_RADII_RATIO=1):
    EVENT_WEIGHT = 2
    ALL_EVENTS = []
    #events will be along a circular peripheri
    #the circle centre will be aligned with the grid center
    #circle radius will be 
    R = ((MAX_SCALE/2)*CIRCLE_RADII_RATIO)
    for i in range(0,NUM_EVENTS):
        cur_id = i
        rand_loc = np.random.rand(1,1).flatten()
        pi = math.acos(-1.0)
        angle = rand_loc[0]*pi*2
        r = Region(MAX_SCALE/2+R*math.cos(angle),MAX_SCALE/2+R*math.sin(angle))
        w = EVENT_WEIGHT
        t1 = np.random.randint(0,MAX_TIME)
        t2 = np.random.randint(0,MAX_TIME)
        tint = TimeInterval(t1,t2)
        ALL_EVENTS.append(Event(cur_id,r,w,tint))
    return ALL_EVENTS

#RANDOM_EVENTS = create_random_events(10)
## create agents:
def create_random_agents(NUM_AGENTS):
    RANDOM_AGENTS = []
    for i in range(0,NUM_AGENTS):
        cur_id = i
        rand_loc = np.random.rand(1,2).flatten()
        agent_rand_loc = np.random.rand(1,2).flatten()
        agent_r = Region(agent_rand_loc[0]*MAX_SCALE,agent_rand_loc[1]*MAX_SCALE)
        john = Agent(cur_id, agent_r,1)
        RANDOM_AGENTS.append(john)
    return RANDOM_AGENTS

def get_rectanglexy_grid(s,d,scale,stepidx,stepsize):
    xy = [0,0]
    idx_to_fix = s//2
    fixed_to = (s%2)*scale
    xy[idx_to_fix] = (fixed_to+(stepidx*stepsize))*scale
    xy[1-idx_to_fix] = d*scale
    return xy[0],xy[1]

def create_gridded_agents(NUM_AGENTS,CIRCLE_RADII_RATIO,RECTANGLE_SCALE=1,SPLIT=1):
    ALL_AGENTS = []
    CENTRE = ((MAX_SCALE/2)*CIRCLE_RADII_RATIO)
    SIDE_SCALE = CENTRE*(math.sqrt(2)*RECTANGLE_SCALE)
    SCALE = (MAX_SCALE/2) - SIDE_SCALE/2
    STEPSIZE = (1.0/SPLIT)
    for i in range(0,NUM_AGENTS):
        cur_id = i
        D = np.random.rand(1,1).flatten()[0]
        SIDE = np.random.randint(low=0, high=4, size=(1,))
        STEPIDX = np.random.randint(low=0,high=SPLIT+1,size=(1,))
        #print(SIDE[0])
        X,Y = get_rectanglexy_grid(SIDE[0],D,SIDE_SCALE,STEPIDX,STEPSIZE)
        #print(X,Y)
        #print(X+SCALE,Y+SCALE)
        agent_r = Region(X+SCALE,Y+SCALE)
        john = Agent(cur_id, agent_r,1)
        ALL_AGENTS.append(john)
    return ALL_AGENTS

random.seed(123)
np.random.seed(42)

all_neighbors = []

def calc_average_degree(num_agents):
    agent_degrees = []
    for i in range(0,num_agents):
        agent_degrees.append(0)
    for i in range(len(all_neighbors)):
        agent_degrees[all_neighbors[i][0].id] = agent_degrees[all_neighbors[i][0].id]+1
        agent_degrees[all_neighbors[i][1].id] = agent_degrees[all_neighbors[i][1].id]+1
    ss = 0.0
    for i in range(0,num_agents):
        agent_degrees[i] = agent_degrees[i]/2
        ss = ss + agent_degrees[i]
    return ss/num_agents

def create_neighbors(agent_a,agent_b,P):
    #D = (agent_a.region).dis(agent_b.region)
    q = random.uniform(0,1)
    if q<=P:
        all_neighbors.append([agent_a,agent_b])
        if agent_a.id!=agent_b.id:
            all_neighbors.append([agent_b,agent_a])
    return


def isneighbor(agent_a,agent_b):
    return [agent_a,agent_b] in all_neighbors

# for a in RANDOM_AGENTS:
#     print(a,john,isneighbor(a,john))

import sys

PSPACE = 15
TSPACE = 50
DELTA = 2 # pore change kora jaite pare
TOT_SCENARIO = 100
NUM_AGENTS = int(sys.argv[1])
P = float(sys.argv[2])
THRESH = float(sys.argv[3])
GRIDS = float(sys.argv[4])
NUM_EVENTS = int(sys.argv[5])
SIM_TIME = 30
THETA = 0.9 #pore change kora jaite pare
#update_migration_from_peer

grid = 4

#df = pd.DataFrame(columns=['P','T','SCENE_IDX','MIGRATE','N','GRIDS','NUM_EVENTS'])

all_scenario = []

tot_migrant_percent = 0.0

for scenario_idx in range(0,TOT_SCENARIO):

    RANDOM_EVENTS = create_same_events_circle(NUM_EVENTS,0.9)

    RANDOM_AGENTS = create_gridded_agents(NUM_AGENTS,0.9,SPLIT=GRIDS)

    all_neighbors = []

    for i in range(0,len(RANDOM_AGENTS)):
        for j in range(i,len(RANDOM_AGENTS)):
            create_neighbors(RANDOM_AGENTS[i],RANDOM_AGENTS[j],P)

    ts = []
    people_migrated = []

    for agent in RANDOM_AGENTS:
        agent.reset_agent()

    for t in range(0,SIM_TIME):

        #print("\nAt time ",t)
        for john in RANDOM_AGENTS:
            cur_event_set = []
            for e in RANDOM_EVENTS:
                if e.happened(t):
                    cur_event_set.append(e)
            risk = riskfunc(john,cur_event_set,DELTA)
            john.update_fear(risk,THETA)

        neighbor_intentions = []
        neighbor_sizes = []
        ##synchronous, so everyone updates after interaction
        for i in range(0,len(RANDOM_AGENTS)):
            cur_agent = RANDOM_AGENTS[i]
            #print('agent',cur_agent.id,'has intention:',cur_agent.intention,'migration:',cur_agent.migration,'before peer interaction')
            neighbor_effect = 0
            tot_neighbor = 1
            for j in range(0,len(RANDOM_AGENTS)):
                if i==j:
                    continue
                next_agent = RANDOM_AGENTS[j]
                if isneighbor(cur_agent,next_agent):
                    neighbor_effect = neighbor_effect+max(next_agent.intention,next_agent.migration)
                    tot_neighbor = tot_neighbor + 1
            neighbor_intentions.append(neighbor_effect)
            neighbor_sizes.append(tot_neighbor)

        ##update
        #print('')
        migrated = 0
        for i in range(0,len(RANDOM_AGENTS)):
            cur_agent = RANDOM_AGENTS[i]
            cur_agent.update_migration_from_peer(neighbor_intentions[i]/neighbor_sizes[i],THRESH)
            #print('agent',cur_agent.id,'has intention:',cur_agent.intention,'migration:',cur_agent.migration,'after peer interaction')
            migrated = migrated + cur_agent.migration

        ts.append(t)
        people_migrated.append(migrated)

    for i in range(len(people_migrated)):
        people_migrated[i] = people_migrated[i]/len(RANDOM_AGENTS)

    tot_migrant_percent = tot_migrant_percent + people_migrated[-1]
    new_row = {'P':P,'T':THRESH,'SCENE_IDX':scenario_idx,'MIGRATE':people_migrated[-1],'N':NUM_AGENTS,'GRIDS':GRIDS,'NUM_EVENTS':NUM_EVENTS}
    all_scenario.append(new_row)
    #df.to_csv(f'sim-results-beta/simulation-stat-n-{NUM_AGENTS}-p-{P}-t-{THRESH}-g-{GRIDS}-ne-{NUM_EVENTS}-results.csv',index=False)
    
    # df.to_csv('sim-results-beta/simulation-stat-n-'+str(NUM_AGENTS)+'-p-'+str(P)+'-t-'+str(THRESH)+'-g-'+str(GRIDS)+'-ne-'+str(NUM_EVENTS)+'-results.csv',index=False)

print('Done')
df = pd.DataFrame.from_dict(all_scenario)
df.to_csv('sim-results-beta/simulation-stat-n-'+str(NUM_AGENTS)+'-p-'+str(P)+'-t-'+str(THRESH)+'-g-'+str(GRIDS)+'-ne-'+str(NUM_EVENTS)+'-results.csv',index=False)