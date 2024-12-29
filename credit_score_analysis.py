import privugger as pv
import arviz as az
import numpy as np


def credit_score_analysis(credit_scores):
    def z_score(score):
        return (score - np.mean(credit_scores)) / np.std(credit_scores)
    
    def is_significant_outlier(score):
        return abs(z_score(score)) > 3

    median = np.median(credit_scores)
    num_credit_scores = len(credit_scores)
    deviations = [0] * num_credit_scores
    for i in range(num_credit_scores):
        deviations[i] = abs(credit_scores[i] - median)

    median_absolute_deviation = np.median(deviations)
    minimum_threshold = (max(credit_scores) - min(credit_scores)) / 20
    threshold = max(median_absolute_deviation * 2.5, minimum_threshold)

    outliers = [0] * num_credit_scores
    significant_outliers = 0
    for i in range(num_credit_scores):
        deviation = abs(credit_scores[i] - median)
        if deviation > threshold:
            outliers[i] = 1
            if is_significant_outlier(credit_scores[i]):
                significant_outliers = significant_outliers + 1

    return np.array([sum(outliers), significant_outliers, median])


credit_scores = pv.Uniform("credit_scores", lower=0, upper=10, num_elements=20)
dataset = pv.Dataset(input_specs=[credit_scores])
program = pv.Program(
    name="output",
    dataset=dataset,
    output_type=pv.Float,
    function=credit_score_analysis,
)

trace: az.InferenceData = pv.infer(
    program, cores=4, draws=10_000, method=pv.Method.PYMC, use_new_method=True
)

print(trace["posterior"])
print(trace["posterior"]["return - 27"][0])
