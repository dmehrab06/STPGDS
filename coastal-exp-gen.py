import numpy as np
p_space = [0.07,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#low_p_space = [0.07,0.1,0.15,0.2,0.25,0.3,0.4,0.5]
#low_p_space = [0.05,0.1,0.2,0.3,0.5]
hi_p_space = [0.9]
#n_space = [10,20,30,50,80,100]
n_space = [50,100]
e_space = [10,50]
#t_space = np.linspace(0,1,50)
#t_space = [0,0.1,0.2,0.3,0.4,0.5,0.54,0.58,0.62,0.66,0.68,0.7,0.72,0.74,
#          0.75,0.76,0.78,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.00]

t_space = np.linspace(0,1,50)
#t_space = [0,1/16,1/8,1/4,1/2,3/5,5/8,2/3,3/4,0.76,0.78,4/5,0.81,5/6,6/7,7/8,8/9,9/10]

g_space = [1,2,4]

for n in n_space:
    for p in hi_p_space:
        for t in t_space:
            for g in g_space:
                for e in e_space:
                    print('sbatch setting-n-p-t-grid.sbatch',n,p,t,g,e)
