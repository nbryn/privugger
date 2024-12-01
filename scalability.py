import numpy as np
import time
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import privugger as pv


def dp_mean(ages):
    mean = sum(ages) / len(ages)
    epsilon = 0.1
    delta = 100 / len(ages)  # Assumes ages are in the interval [0-100]
    nu = np.random.laplace(loc=0.0, scale=delta / epsilon)
    dp_mean = mean + nu
    return dp_mean

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

def anomaly_detection(sensor_readings):
    #sensor_readings = [1, 1, 100, 1, 1, 1, 300]

    def is_severe_anomaly(deviation, threshold):
        return deviation > 2 * threshold

    # Step 1: Compute the median of the sensor readings
    median = np.median(sensor_readings)

    # Step 2: Compute the Median Absolute Deviations (MAD)
    num_readings = len(sensor_readings)
    deviations = [0] * num_readings
    for i in range(num_readings):
        deviations[i] = abs(sensor_readings[i] - median)

    median_absolute_deviation = np.median(deviations)
    threshold = median_absolute_deviation * 2.5

    # Step 3: Identify anomalies and classify severity
    anomalies = [(0, False)] * num_readings
    anomaly_count = 0

    for i in range(num_readings):
        deviation = abs(sensor_readings[i] - median)
        if deviation > threshold:
            anomalies[anomaly_count] = (i, is_severe_anomaly(deviation, threshold))
            anomaly_count += 1

    # Return only the relevant portion of the preallocated list
    return anomalies[:anomaly_count]


def measure_execution_time(num_elements, runs=10):
    times = []
    for _ in tqdm(range(runs), desc=f"Processing {num_elements} elements"):
        ages = pv.Uniform("ages", lower=0, upper=100, num_elements=num_elements)
        ds = pv.Dataset(input_specs=[ages])
        pv.concatenate
        program = pv.Program(
            "output", dataset=ds, output_type=pv.Float, function=anomaly_detection
        )
        start_time = time.time()
        _ = pv.infer(
            program, cores=4, draws=10_000, method=pv.Method.PYMC, use_new_method=True
        )
        end_time = time.time()
        times.append(end_time - start_time)
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(
        f"Finished {num_elements} elements: Mean Time = {mean_time:.4f}s, Std Dev = {std_time:.4f}s"
    )
    return mean_time, std_time


# Input sizes to test
input_sizes = [25, 50, 100, 200, 400]
results = []

print("Starting scalability experiment...")
for size in tqdm(input_sizes, desc="Overall Progress"):
    mean_time, std_time = measure_execution_time(size, 2)
    results.append((size, mean_time, std_time))

# Results Visualization
print("Experiment complete. Generating plots...")
df = pd.DataFrame(results, columns=["Input Size", "Mean Time", "Std Dev"])

# Combined Plot with Shaded Area for Standard Deviation
plt.figure(figsize=(8, 6))
plt.plot(
    df["Input Size"],
    df["Mean Time"],
    marker="o",
    label="Mean Execution Time",
    color="blue",
)
plt.fill_between(
    df["Input Size"],
    df["Mean Time"] - df["Std Dev"],
    df["Mean Time"] + df["Std Dev"],
    color="blue",
    alpha=0.2,
    label="±1 Std Dev",
)
plt.title("Execution Time vs Input Size")
plt.xlabel("Input Size")
plt.ylabel("Time (seconds)")
plt.xticks(df["Input Size"])  # Set x-ticks to the input sizes actually used
plt.legend()
plt.grid()
plt.show()
