import sys
import os
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

# Should we handle programs like this in the research project?
# Problems:
# - Hard to map AST nodes to line numbers
#  - (Most) Node don't have a that info
# Ideas:
# - Treat intermediate computations as 'programs' 
#   - Insert return and map to PyMC
#   - Retrieve line numbers after return has been inserted
def problem(ages):
    result = None
    if ages[0] < 35:
        subset = ages[:20]
        result = subset.sum() / subset.size
        
    if ages[0] < 40:
        subset = ages[:40]
        result = subset.sum() / subset.size
    
    else:
        subset = ages[:60]
        result = subset.sum() / subset.size
           
    return result



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
    if ages[0] < 35:
        subset = ages[:20]
        avg = subset.sum() / subset.size
        return avg
    
    if ages[0] < 40:
        subset = ages[:40]
        avg = subset.sum() / subset.size
        return avg + 2
    
    elif ages[0] < 45:
        subset = ages[:60]
        avg = subset.sum() / subset.size
        return avg
            
    return ages.sum() / ages.size

""" basic_model = pm.Model()
with basic_model:
    x = pm.Normal("t", mu=0, sd=1, shape=100)
    print(x) """

ages = pv.Normal('ages', mu=35, std=2, num_elements=100)
ds = pv.Dataset(input_specs=[ages])
program = pv.Program('output',
                     dataset=ds,
                     output_type=pv.Float,
                     function=analyze_works)

program.add_observation('output==44', precision=0.1)

trace = pv.infer(program,
                 cores=4,
                 draws=50,
                 method='pymc3')



# plot the inferred distribution of the output

#print(line_number)
az.plot_posterior(trace, var_names=['output'],
                    hdi_prob=.95, point_estimate='mode')
plt.show()
