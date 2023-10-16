import sys
import os
import privugger as pv
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import warnings

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
    if len(ages) < 5:
        result = ages.sum()
    
    elif len(ages) < 10:
        result = ages[0]
    
    else:
        result = (ages.sum()) / (ages.size)
            
    return result



# Handle AST node with return that isn't a 'top level' node
def analyze_in_progress(ages):
    if len(ages) < 5:
        total = ages.sum()
        return total
    
    if len(ages) < 10:
        if len(ages) < 12:
            first = ages[0]
            return first
        else:
            second = ages[1]
            return second
    
    elif len(ages) < 15:
        third = ages[2]
        return third
            
    return ages.sum() / ages.size




def analyze_works(ages):
    if len(ages) < 5:
        total = ages.sum()
        return total
    
    if len(ages) < 10:
        first = ages[0]
        return first
    
    elif len(ages) < 15:
        second = ages[1]
        return second
            
    return ages.sum() / ages.size



ages = pv.Normal('ages', mu=35, std=2, num_elements=100)
ds = pv.Dataset(input_specs=[ages])
program = pv.Program('output',
                     dataset=ds,
                     output_type=pv.Float,
                     function=analyze_works)

program.add_observation('output==44', precision=0.1)

traces = pv.infer(program,
                 cores=4,
                 draws=50,
                 method='pymc3')

# plot the inferred distribution of the output
for (line_number, trace) in traces:
    print(line_number)
    az.plot_posterior(trace, var_names=['output'],
                      hdi_prob=.95, point_estimate='mode')
    plt.show()
