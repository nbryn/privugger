import privugger as pv
import arviz as az
import numpy as np


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


sensor_readings = pv.Uniform("sensor_readings", lower=0, upper=10, num_elements=20)
dataset = pv.Dataset(input_specs=[sensor_readings])
program = pv.Program(
    name="output",
    dataset=dataset,
    output_type=pv.Float,
    function=anomaly_detection,
)

trace: az.InferenceData = pv.infer(
    program, cores=4, draws=10_000, method=pv.Method.PYMC, use_new_method=True
)

print(trace["posterior"])
#print(trace["posterior"]["return - 30"])
