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
warnings.simplefilter(action="ignore", category=FutureWarning)

# move to previouse directory to access the privugger code
sys.path.append(os.path.join("../../"))

def avg(ages):
    return (ages.sum()) / (ages.size)

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


# This should be mapped to what is shown in temp_pymc
def masking(ages):
    if ages[0] < 35:
        subset1 = ages[:20]
        avg1 = subset1.sum() / subset1.size
        return avg1

    if ages[0] < 40:
        subset2 = ages[:40]
        avg2 = subset2.sum() / subset2.size
        return avg2

    elif ages[0] < 45:
        subset3 = ages[:60]
        avg3 = subset3.sum() / subset3.size
        return avg3

    return ages.sum() / ages.size


def temp_pymc(ages):
    with pm.Model() as model:
        avg1 = pm.Deterministic("avg1", pm.math.sum(ages[:20]) / 20)
        avg2 = pm.Deterministic("avg2", pm.math.sum(ages[:40]) / 40)
        avg3 = pm.Deterministic("avg3", pm.math.sum(ages[:60]) / 60)
        avg_all = pm.Deterministic("avg_all", pm.math.sum(ages) / len(ages))

        age = ages[0]
        condition1 = pm.math.lt(age, 35)
        condition2 = pm.math.lt(age, 40)
        condition3 = pm.math.lt(age, 45)

        output = pm.math.switch(
            condition1,
            avg1,
            pm.math.switch(condition2, avg2, pm.math.switch(condition3, avg3, avg_all)),
        )
        pm.Deterministic("output", output)

    return model

def ages_dp_pymc():
    with pm.Model() as model:
        ages = pm.Uniform('ages', lower=0, upper=100, size=100)
        avg = pm.Deterministic('avg', pm.math.sum(ages) / ages.shape)
        epsilon = pm.Deterministic('epsilon', 0.1)
        delta = pm.Deterministic('delta', 100 / ages.shape)
        nu = pm.Laplace('nu', mu=0, b=delta / epsilon)
        dp_avg = pm.Deterministic('dp_avg', avg + nu)

    return model

# Data representing a uniform distribution over the values 0, 1, 2, 3
""" data = np.array([0, 1, 2, 3])

# Create subplots with 1 row and 5 columns
fig, axs = plt.subplots(1, 5, figsize=(15, 3), sharey=True)

# Plotting histograms
counter = 0
for ax in axs:
    ax.hist(data, bins=[-0.5, 0.5, 1.5, 2.5, 3.5], weights=np.ones_like(data) / len(data),
            edgecolor='black', linewidth=1.2)

    # Customizing each subplot
    ax.set_title("Index: " + str(counter))
    ax.set_xlabel("Values")
    counter += 1

# Set a common ylabel for the entire figure
axs[0].set_ylabel("Probability")

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show() """


# --- A simple program for each transformation rule ---

# Works
# TODO: Also handle [1,2,3].sum()?
def call_example():
    ages = [1, 2, 3] 
    test = ages.sum()
    return test

# Works
# TODO: Ensure we can handle 'expression if condition else expression'
def if_example():
    x = 5
    y = 0
    if x > 10:
        y = 20 
    else: 
        y = 30

    return y
    
# TODO: Ensure that this work after implementing neural network example
def ages_dp(ages):
    ages0 = ages[0]
    avg = ages.sum() / ages.size
    epsilon = 0.1
    delta = 100 / ages.size # assumes ages are in the interval [0-100]
    nu = np.random.laplace(loc=0.0, scale=delta / epsilon)
    dp_avg = avg + nu

    return dp_avg

# TODO: Try other variations of loops
# range(10)
# range(0, 1)
# while
def masking2(ages):
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


# TODO: Variables like subset1 should only exist if condition is true
# TODO: Variables outside if should only exist if condition is not true (and if has return)
def masking1(ages):
    h = 10
    if h > 35:
        t = 2
        subset1 = ages[:20]
        ht = [0]
        ht1 = []
        avg1 = subset1.sum() / subset1.size
        return avg1

    mt = 2
    return ages.sum() / ages.size

# TODO: Try to model this in PyMC with and without using switch


def temp(ages):
    # A
    if ages[0] < 10:
        a = ages[0]
        return a
    
    # B
    if ages[0] < 50:
        b = ages[1]
        return b

    # C
    c = ages[2]
    return c

def neural_network(ages):
    # Activation function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Inputs for XOR problem
    X = np.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]])

    # Trained weights
    weights_input_hidden = np.array([[ 5.07377189,  6.31596517, -7.34522284, -6.91238663],
                                    [ 5.09765788, -5.87056886,  7.4695342 , -7.20017149]])

    weights_hidden_output = np.array([[-10.05563681],
                                    [ 10.09224415],
                                    [-10.11382543],
                                    [ 10.15214779]])

    # Forward propagation using the trained weights
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    final_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    final_layer_output = sigmoid(final_layer_input)

    #print("Output using pre-trained weights: ")
    #print(final_layer_output)


# TODO: Handle function calls


# TODO: Doesn't work
def next():
    temp = []
    output = [1, 2, 3]
    output[len(temp)] = 0

    return output


ages = pv.Uniform("ages", lower=0, upper=100, num_elements=20)
ds = pv.Dataset(input_specs=[ages])
program = pv.Program("output", dataset=ds, output_type=pv.Float, function=neural_network)
program.add_observation("output==44", precision=0.1)

trace: az.InferenceData = pv.infer(program, cores=4, draws=10_000, method="pymc3", use_new_method=True)

print(trace["posterior"])
#az.plot_posterior(trace, var_names=['return - 13'], hdi_prob=.95)

#mi_avg = pv.mi_sklearn(trace, var_names=["ages0 - 2", "avg - 3"])
#mi_dp_avg = pv.mi_sklearn(trace, var_names=["ages0 - 2", "dp_avg - 7"])

#print(mi_avg[0])
#print(mi_dp_avg[0])
#print(len(trace["posterior"]["ages"][0][0]))
#print(len(trace["posterior"]["subset1 - 3"][0][0]))
#print(len(trace["posterior"]["subset2 - 8"][0][0]))
#print(len(trace["posterior"]["subset3 - 13"][0][0]))

#print(trace["posterior"]["subset3 - 13"][0][0])
#print(trace["posterior"]["subset3 - 13"][0][1])

print(trace.posterior["ages"][0][0])
print(trace.posterior["output - 2"][0][0])
print(trace.posterior["return - 13"][0][0])

plt.show()
