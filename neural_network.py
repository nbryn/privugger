import privugger as pv
import numpy as np
import arviz as az


def neural_network(votes):
    # Activation functions
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def relu(x):
        return x * (x > 0)

    # Adjust weights to account for input size
    first_layer_weights = np.ones((len(votes), 4))
    second_layer_weights = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )

    third_layer_weights = np.array([1.0, 1.0])

    # Forward propagation
    first_layer_output = sigmoid(np.dot(votes, first_layer_weights))
    second_layer_output = sigmoid(np.dot(first_layer_output, second_layer_weights))
    third_layer_output = sigmoid(np.dot(second_layer_output, third_layer_weights))

    final_layer_input = np.dot(third_layer_output, third_layer_weights)
    result = relu(final_layer_input)

    return 1 if result >= 0.5 else 0

input = pv.DiscreteUniform("input", lower=0, upper=1, num_elements=20)
dataset = pv.Dataset(input_specs=[input])

program = pv.Program(
    name="output",
    dataset=dataset,
    output_type=pv.Float,
    function=neural_network,
)

trace: az.InferenceData = pv.infer(
    program, cores=4, draws=10_000, method=pv.Method.PYMC, use_new_method=True
)

print(trace["posterior"])
