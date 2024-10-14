import pymc as pm
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


# Report
# Why is this useful?
# Simple example: Privacy
# Complicated example: Neural Network
# Scalability: Neural Network


# All variables must be declared at the outermost (function) scope?
# - Maybe we dont need the above restriction?
# Maximum one return at the end of program
# Remember to discuss why in the report
def temp(ages):
    # A
    a
    if ages[0] < 10:
        a = ages[0]

        #unique variable = a
    
    # B
    if ages[0] < 50:
        b = ages[1]
        # unique b

    # C
    c = ages[2]
    return c






# We should 
def temp2(ages):
    def f(t):
        h = 5 # Should be in the trace. Keep track of number of invocations for f
        g = 6
        return t + h + g

    th = f(1)
    th1 = f(2)
    
    return th + th1




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

    return final_layer_output

# TODO: Get this to work
def neural_network2(input):
    # Activation function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    weights = [2, 5, 7]

    # Compute output -> logistic(Σ_i x_i·w_i)
    output = np.dot(input, weights)
    final_output = sigmoid(output)

    return final_output