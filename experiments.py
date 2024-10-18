import sys
import os
import math
import privugger as pv
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import warnings
import pymc as pm

# disable FutureWarnings to have a cleaner output
warnings.simplefilter(action="ignore", category=FutureWarning)

# move to previouse directory to access the privugger code
sys.path.append(os.path.join("../../"))


def avg(ages):
    return (ages.sum()) / (ages.size)


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
        ages = pm.Uniform("ages", lower=0, upper=100, size=100)
        avg = pm.Deterministic("avg", pm.math.sum(ages) / ages.shape)
        epsilon = pm.Deterministic("epsilon", 0.1)
        delta = pm.Deterministic("delta", 100 / ages.shape)
        nu = pm.Laplace("nu", mu=0, b=delta / epsilon)
        dp_avg = pm.Deterministic("dp_avg", avg + nu)

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
# TODO: Also handle sum([1,2,3])?
def call_example():
    ages = [1, 2, 3]
    test = sum(ages)
    return test


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


def neural_network(ages):
    # Activation function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Inputs for XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Trained weights
    weights_input_hidden = np.array(
        [
            [5.07377189, 6.31596517, -7.34522284, -6.91238663],
            [5.09765788, -5.87056886, 7.4695342, -7.20017149],
        ]
    )

    weights_hidden_output = np.array(
        [[-10.05563681], [10.09224415], [-10.11382543], [10.15214779]]
    )

    # Forward propagation using the trained weights
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    final_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    final_layer_output = sigmoid(final_layer_input)

    return final_layer_output


# TODO: Confirm trace look as expected. neural_network2 afterwards
def ages_dp(ages):
    ages0 = ages[0]
    avg = sum(ages) / len(ages)
    epsilon = 0.1
    delta = 100 / len(ages)  # assumes ages are in the interval [0-100]
    nu = np.random.laplace(loc=0.0, scale=delta / epsilon)
    dp_avg = avg + nu

    return dp_avg


# TODO: Doesn't work
def next():
    temp = []
    output = [1, 2, 3]
    output[len(temp)] = 0

    return output


# This works
def neural_network2(input):
    # Activation function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    weights = [2, 5, 7]

    # Compute output -> logistic(Σ_i x_i·w_i)
    output = np.dot(input, weights)
    final_output = sigmoid(output)

    return final_output


# Take a look at this if time
def temp2(ages):
    def f(t):
        h = 5  # Should be in the trace. Keep track of number of invocations for f
        g = 6
        return t + h + g

    th = f(1)
    th1 = f(2)
    return th + th1


def naive_k_anonymity(ages):
    k = 2
    if len(ages) == 0:
        return ages

    num_attrs = len(ages[0])
    num_records = len(ages)
    for i in range(num_attrs):
        target_attr = num_attrs - 1 - i
        if not target_attr == num_attrs:
            for j in range(num_records):
                ages[j][target_attr] = -1

        for i in range(num_records):
            k_prime = 0
            for j in range(num_records):
                if ages[i] == ages[j]:
                    k_prime = k_prime + 1

            if k_prime < k:
                break

        if not k_prime < k:
            return ages

    return ages

# TODO: All instances of 't' should not be set to the same final value
# IE if the first else has been hit, then t on line 247 should be 200 otherwise just 5
# atm all instances of t in the loop is set to 100 in the output
def TEMP2(ages):
    t = 5
    for i in range(10):
        if i == 9:
            t = 100
     
        else:
            t = 200
            
            if t == 200:
                t = 300
            else:
                t = 500

    return t


# TODO: Break doesn't work
# IE: this program returns 9
def TEMP(ages):
    t = 5
    for i in range(10):
        t = i
        if i == 8:
            break


    return t



ages = pv.Uniform("ages", lower=0, upper=100, num_elements=20)
ds = pv.Dataset(input_specs=[ages])
program = pv.Program("output", dataset=ds, output_type=pv.Float, function=TEMP)
program.add_observation("output==44", precision=0.1)

trace: az.InferenceData = pv.infer(
    program, cores=4, draws=10_000, method=pv.Method.PYMC, use_new_method=True
)

print(trace["posterior"])
# az.plot_posterior(trace, var_names=['return - 13'], hdi_prob=.95)

# mi_avg = pv.mi_sklearn(trace, var_names=["ages0 - 2", "avg - 3"])
# mi_dp_avg = pv.mi_sklearn(trace, var_names=["ages0 - 2", "dp_avg - 7"])

# print(mi_avg[0])
# print(mi_dp_avg[0])
# print(len(trace["posterior"]["ages"][0][0]))
# print(len(trace["posterior"]["subset1 - 3"][0][0]))
# print(len(trace["posterior"]["subset2 - 8"][0][0]))
# print(len(trace["posterior"]["subset3 - 13"][0][0]))

# print(trace["posterior"]["subset3 - 13"][0][0])
# print(trace["posterior"]["subset3 - 13"][0][1])

# print(trace.posterior["ages"][0][0])
# print(trace.posterior["output - 2"][0][0])
# print("ACTUAL: ")
# print(neural_network([]))
# print("PYMC: ")
print(trace.posterior["return - 28"][0][0])
print(len(trace.posterior["return - 28"][0][0]))


plt.show()
