import sys
import os
import math
import privugger as pv
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import warnings
import pymc3 as pm

# disable FutureWarnings to have a cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

# move to previouse directory to access the privugger code
sys.path.append(os.path.join("../../"))

# Handle AST node with return that isn't a 'top level' node
def analyze_in_progress(ages):
    if ages[0] < 35:
        subset = ages[:20]
        avg = subset.sum() / subset.size
        return avg
    
    if ages[0] < 40:
        if ages[0] < 42:
            subset = ages[:40]
            avg = subset.sum() / subset.size
            return avg
        else:
            subset = ages[:50]
            avg = subset.sum() / subset.size
            return avg
    
    elif ages[0] < 45:
        subset = ages[:60]
        avg = subset.sum() / subset.size
        return avg
            
    return ages.sum() / ages.size


def analyze_works(ages):
    first = ages[5:10].sum()
    if ages[0] < 45:
        subset = ages[:20]
        avg = subset.sum() / subset.size
        return avg
    
    if ages[0] < 40:
        subset = ages[:40]
        avg = subset.sum() / subset.size
        return avg
    
    elif ages[0] < 45:
        subset = ages[:60]
        avg = subset.sum() / subset.size
        return avg
    
    second = ages[10:20].sum()
    return ages.sum() / ages.size

def masking(ages):
     output = ages
     for i in range(len(ages)):
         if 0 <= ages[i] < 25:
             output[i] = 0
         if 25 <= ages[i] < 50:
             output[i] = 1
         if 50 <= ages[i] < 75:
             output[i] = 2
         if 75 <= ages[i]:
             output[i] = 3
     
     return output
 

ages = pv.Uniform('ages', lower=0, upper=100, num_elements=5)
ds = pv.Dataset(input_specs=[ages])
program = pv.Program('output',
                     dataset=ds,
                     output_type=pv.Float,
                     function=masking)

program.add_observation('output==44', precision=0.1)

trace: az.InferenceData = pv.infer(program,
                 cores=4,
                 draws=10_000,
                 method='pymc3')

print(trace.posterior.data_vars['return'][0])
#temp = trace.posterior.data_vars['avg - 5'][0] 
#print(temp)
#print(len(temp))
#trace.posterior.data_vars['avg - 5'][0] = [x for x in temp if not math.isnan(x) ]

""" az.plot_posterior(trace, var_names=['avg - 5'],
                    hdi_prob=.95, point_estimate='mode') """

""" az.plot_posterior(trace, var_names=['return - 19'],
                    hdi_prob=.95, point_estimate='mode') """


plt.show()
