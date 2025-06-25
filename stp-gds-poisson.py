import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time


class Region:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def shift_by(self,dx,dy):
        self.x = self.x+dx
        self.y = self.y+dy
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

def calc_intention(fear,Q=30,v=0.08):
    return 1.0/(1.0+Q*math.exp(-v*fear))

class Agent:
    def __init__(self, idx, r, beta):
        self.id = idx
        self.region = r
        self.beta = beta #(flee bias beta between [0,1])
        self.fear = 0 #(>0)
        self.intention = calc_intention(self.fear)
        self.migration = 0
        self.migration_time = -1
        self.neighbor_history = []
        
    def update_fear(self,risk,theta=0):
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
        self.migration = 0
        self.migration_time = -1
        self.neighbor_history = []

    def update_migration_from_peer(self,peer_effect,gamma,THRESH,gamma_2=None):
        if gamma_2 is None:
            gamma_2 = (1-gamma)
        tot_weight = gamma*peer_effect+gamma_2*self.fear
        if (tot_weight>=THRESH) or self.migration==1:
            self.migration = 1
            if self.migration_time==-1:
                self.migration_time = len(self.neighbor_history)
        else:
            self.migration = 0
            
    def __str__(self):
        all_info = ('agent id:',str(self.id),'current state:',str(self.migration),'loc:',str(self.region.x),',',str(self.region.y))
        return ' '.join(all_info)

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

def riskfunc(a,E,alpha):
    tot = 0.0
    for event in E:
        #dis = ((a.region).dis(event.region,dis_method='euclidean'))**delta
        dis = np.exp(-(alpha*(a.region).dis(event.region,dis_method='euclidean')))
        #dis = dis+EPS
        cur_fear = (a.beta)*(event.weight)*dis
        tot = tot+cur_fear
    return tot
            

def createplot(Agents,Events,ax,Zoom_Factor=2,Agent_Size=5):
    OFFSET = 0.1
    for a in Agents:
        ax.plot(a.region.x,a.region.y,marker='o',color='blue',markersize=2,
                alpha=0.4)
    for e in Events:
        ax.plot(e.region.x,e.region.y,marker='X',color='red',
                markersize=int(e.weight)*Zoom_Factor)
        #ax.annotate(str(e.event_id), xy =(e.region.x,e.region.y), 
        #     xytext =(e.region.x+OFFSET, e.region.y+OFFSET))
    ax.set_xlim([0,MAX_SCALE])
    ax.set_ylim([0,MAX_SCALE])
    
def create_neighbor_plot(Agents,G,ax,Agent_Size=5):
    for a in Agents:
        for b in Agents:
            if a.id in G.neighbors(b.id):
                ax.plot([a.region.x,b.region.x],[a.region.y,b.region.y],linestyle='--',linewidth=0.5,alpha=0.2,color='black')
                

def create_discrete_poisson_process(rate, radius, time_horizon, delta_t=1):
    event_times = np.arange(0, time_horizon, delta_t)
    num_events = np.random.poisson(rate * delta_t, len(event_times))
    all_events = []
    event_id = 0
    for tidx,t in enumerate(event_times):
        event_at_t = num_events[tidx]
        #print(event_at_t)
        for e in range(event_at_t):
            angle = np.random.uniform(0, 2 * np.pi)  # Angle uniformly from [0, 2*pi]
            radial_distance = radius * np.sqrt(np.random.uniform(0, 1))  # Radial distance uniformly within the circle
            # Convert polar coordinates to Cartesian coordinates
            x = radial_distance * np.cos(angle)
            y = radial_distance * np.sin(angle)
            reg = Region(x,y)
            tint = TimeInterval(t,t)
            all_events.append(Event(event_id,reg,1,tint))
            event_id = event_id + 1
    return all_events

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

def calc_exp_frac(a,r,tau,gamma,rate,deg,gamma_2=None):
    if gamma_2 is None:
        gamma_2 = (1-gamma)
    ar = a*r
    frac_1_nom = (ar**2)*tau - 2*gamma_2*rate*(1-np.exp(-ar)*(1+ar))
    return max(0,min(1,frac_1_nom/((ar**2)*gamma*deg)))

def calc_exp_frac_2(a,r,tau,gamma,rate,deg,gamma_2=None):
    if gamma_2 is None:
        gamma_2 = (1-gamma)
    frac_nom = tau - 2*(gamma_2)*rate
    return max(0,min(1,frac_nom/(gamma*deg)))

def adjust_legend(ax,lgd_params):
    #LEGEND_PARAMS = {'hlen':1,'bpad':0.2,'lspace':0.2,'htxtpad':0.2,'baxpad':0.2,'cspace':0.2,'ncol':2,'ecolor':'black','size':12,'alpha':0.2}
    ax.legend(loc="best", fancybox=True, handlelength=lgd_params['hlen'], borderpad=lgd_params['bpad'], labelspacing=lgd_params['lspace'],
                  handletextpad = lgd_params['htxtpad'], borderaxespad = lgd_params['baxpad'], columnspacing = lgd_params['cspace'],
                  ncol=lgd_params['ncol'], edgecolor=lgd_params['ecolor'], frameon=True, framealpha=lgd_params['alpha'], shadow=False, prop={'size': lgd_params['size']})
    
def evaluate_results(agents,G,alpha,radius,tau,gamma,rate):
    DEL = 0
    we = 0
    they = 0
    tot = 0
    sq_error_our = 0
    sq_error_naive = 0
    for aidx,a in enumerate(agents):
        if (G.degree[aidx]>0):
            if a.migration_time==-1:
                continue
            theory_val = calc_exp_frac(alpha,radius,tau,gamma,rate,G.degree[aidx],gamma_2=None)
            fraction_then = a.neighbor_history[a.migration_time+DEL]
            sq_error_our = sq_error_our + (fraction_then-theory_val)*(fraction_then-theory_val)
            sq_error_naive = sq_error_naive + (min(1,tau/(G.degree[aidx]))-fraction_then)*(min(1,tau/(G.degree[aidx]))-fraction_then)
            better_est = 'our' if abs(fraction_then-theory_val)<abs(min(1,tau/(G.degree[aidx]))-fraction_then) else 'naive'
            tot = tot + 1
            we = we+1 if better_est =='our' else we
            they = they+1 if better_est!='our' else they
    if tot==0:
        return 0.5,-1,-1
    return we/tot,(sq_error_our/tot)**0.5,(sq_error_naive/tot)**0.5

gammas = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
rates = range(1,51,2)
#gammas = [0.05,0.1]
#rates = range(15,21)

gamma_x = []
rate_x = []
acc_our_y = []
mse_our_y = []
mse_naive_y = []

all_dicts = []

GN = str(sys.argv[1])
SEED_START = int(sys.argv[2])
SEED_END = int(sys.argv[3])

st = time.time()

for gamma in gammas:
    for rate in rates:
        ## Simulation Parameters
        SEEDS = range(SEED_START,SEED_END)
        for SEED in SEEDS:
            np.random.seed(SEED)
            N = 100
            P = 0.2
            K = 30
            M = 11
            G = nx.erdos_renyi_graph(N,P,seed=SEED) if GN=='ER' else (nx.barabasi_albert_graph(N,M,seed=SEED) if GN=='BA' else nx.watts_strogatz_graph(N,K,P,seed=SEED))
            MAX_SCALE = 10 # space will be within (0,0) -- (10,10)
            radius = 1     # Radius of the circle
            time_horizon = 45  # Total number of time steps
            delta_t = 1      # Time step size
            alpha = 0.2
            SIM_TIME = 100
            tau = 11

            agents = create_random_agents(N)
            ALL_EVENTS = []
            for a in agents:
                events = create_discrete_poisson_process(rate, radius, time_horizon, delta_t)
                for idx in range(len(events)):
                    events[idx].region.shift_by(a.region.x,a.region.y)
                ALL_EVENTS.append(events)

            ts = []
            people_migrated = []
            fixed = -1

            for a in agents:
                a.reset_agent()
            try:
                for t in range(0,SIM_TIME):
                    for aidx,a in enumerate(agents):
                        a_event_set = ALL_EVENTS[aidx]
                        cur_event_set = []
                        for e in a_event_set:
                            if e.happened(t):
                                cur_event_set.append(e)

                        risk = riskfunc(a,cur_event_set,alpha)
                        a.update_fear(risk,0)

                    temp_neighbor_effects = []

                    for aidx,a in enumerate(agents):
                        N_v = G.neighbors(aidx)
                        peer_effect = sum([agents[v].migration for v in N_v])
                        temp_neighbor_effects.append(peer_effect)

                    migrated = 0    
                    for aidx,a in enumerate(agents):
                        a.update_migration_from_peer(temp_neighbor_effects[aidx],gamma,tau,gamma_2=None)
                        if G.degree[aidx]>0:
                            a.neighbor_history.append(temp_neighbor_effects[aidx]/G.degree[aidx])
                        migrated = migrated + a.migration

                    ts.append(t)
                    people_migrated.append(migrated)
                    if migrated==N and fixed==-1:
                        fixed = t
                acc,mse_our,mse_naive = evaluate_results(agents,G,alpha,radius,tau,gamma,rate)
                print(gamma,rate,acc,mse_our,mse_naive,flush=True)
                #print(gamma,rate,evaluate_results(agents,G,alpha,radius,tau,gamma,rate))
                graph_name = f'ER_N_{N}_P_{P}' if GN=='ER' else (f'BA_N_{N}_M_{M}' if GN=='BA' else f'WS_N_{N}_K_{K}_P_{P}')
                cur_dict = {'gamma':gamma,'rate':rate,'seed':SEED,'acc':acc,'mse_our':mse_our,'mse_naive':mse_naive,'graph':graph_name}
                all_dicts.append(cur_dict)
            except Exception as e:
                print(e)
                continue
pd.DataFrame.from_dict(all_dicts).to_csv(f'Poisson_simulation_results/poisson_{graph_name}_SEED_{SEED_START}_TO_{SEED_END}.csv',index=False)
print('Done')
print('time taken',time.time()-st)