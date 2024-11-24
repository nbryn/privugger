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


def measure_execution_time(num_elements, runs=10):
    times = []
    for _ in tqdm(range(runs), desc=f"Processing {num_elements} elements"):
        ages = pv.Uniform("ages", lower=0, upper=100, num_elements=num_elements)
        ds = pv.Dataset(input_specs=[ages])
        program = pv.Program(
            "output", dataset=ds, output_type=pv.Float, function=dp_mean
        )
        program.add_observation("output==44", precision=0.1)
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
input_sizes = [50, 200, 1000, 2000, 3500]
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
