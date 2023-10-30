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
    if ages[0] < 35:
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


ages = pv.Normal('ages', mu=35, std=2, num_elements=100)
ds = pv.Dataset(input_specs=[ages])
program = pv.Program('output',
                     dataset=ds,
                     output_type=pv.Float,
                     function=analyze_works)

program.add_observation('output==44', precision=0.1)

trace: az.InferenceData = pv.infer(program,
                 cores=4,
                 draws=10_000,
                 method='pymc3')

#print(trace.posterior)
temp = trace.posterior.data_vars['avg - 5'][0] 
print(temp)
print(len(temp))
#trace.posterior.data_vars['avg - 5'][0] = [x for x in temp if not math.isnan(x) ]

az.plot_posterior(trace, var_names=['avg - 5'],
                    hdi_prob=.95, point_estimate='mode')

""" az.plot_posterior(trace, var_names=['return - 19'],
                    hdi_prob=.95, point_estimate='mode') """


plt.show()
