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

def avg_or_sum(ages):
    if len(ages) > 5:
        return ages.sum()
    
    return (ages.sum()) / (ages.size)


ages = pv.Normal('ages', mu=35, std=2, num_elements=100)
ds = pv.Dataset(input_specs=[ages])
program = pv.Program('output',
                     dataset=ds,
                     output_type=pv.Float,
                     function=avg_or_sum)

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
