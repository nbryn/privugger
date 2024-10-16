import pymc as pm
import numpy as np



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

print(neural_network2([1,2,3]))