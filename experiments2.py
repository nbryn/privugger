import pymc3 as pm
import numpy as np

# Define the model
with pm.Model() as model:
    # Define a normal distribution with a specified shape
    normal_dist = pm.Normal('normal_dist', mu=0, sigma=1, shape=100)
    lognormal_dist = pm.Lognormal('lognormal_dist', mu=0, sigma=1)
    print(type(normal_dist))
    print(type(lognormal_dist))

    # Simulate a draw from the distribution (let's say we want 5 samples)
    sample_indices = np.arange(5)  # This is equivalent to taking the first 5 samples for demonstration
    i, j = 0, 2

    # Create a deterministic variable to represent the slice [i:j]
    sliced_samples = pm.Deterministic('sliced_samples', normal_dist[i:j])

    # Use pm.math.sum to sum the sliced samples
    sum_sliced_samples = pm.Deterministic('sum_sliced_samples', pm.math.sum(sliced_samples))

    # Sample from the model
    trace = pm.sample(1000, tune=500, chains=2, return_inferencedata=True)

# Extract the sum of sliced samples from the trace
#sum_sliced_samples_trace = trace['sum_sliced_samples']
#print("Sum of Sliced Samples Trace: ", sum_sliced_samples_trace[:5]) 

print(trace["posterior"])
print(trace["posterior"]["normal_dist"])
print(trace["posterior"]["normal_dist"][0])
print(trace["posterior"]["normal_dist"][0][0])
print(len(trace["posterior"]["normal_dist"]))
print(len(trace["posterior"]["normal_dist"][0]))
print(len(trace["posterior"]["normal_dist"][0][0]))

print("SLICED SAMPLES")

print(len(trace["posterior"]["sliced_samples"][0][0]))




# 1: Require that all variables outside if statements

# 2: Default value if conditions hold
# Discrete 0, Continous 0.0


# Should subset1 be undefined if _ages[0]_ >= 35?
# Or should subset1 always exist since we are quantifying risk?
def example1(ages):
    if ages[0] < 35:
        # A vector of 0 or 0.0
        subset1 = ages[:20]
        avg1 = subset1.sum() / subset1.size
        return avg1

    return 5


# Should reassignment be treated as a new variable?
def example2():
    x = 10
    y = 0
    if x > 5:
        y = 2
    else:
        y = 10
    
    s = 2 + y
    
    return y