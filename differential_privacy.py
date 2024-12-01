import privugger as pv
import arviz as az
import numpy as np
import pymc as pm


def dp_mean(ages):
    mean = sum(ages) / len(ages)
    epsilon = 0.1
    delta = 100 / len(ages)  # Assumes ages are in the interval [0-100]
    nu = np.random.laplace(loc=0.0, scale=delta / epsilon)
    dp_mean = mean + nu

    return dp_mean

def dp_mean_pymc():
    with pm.Model() as model:
        ages = pm.Uniform("ages", lower=0, upper=100, size=100)
        mean = pm.Deterministic("mean", pm.math.sum(ages) / ages.shape)
        epsilon = pm.Deterministic("epsilon", 0.1)
        delta = pm.Deterministic("delta", 100 / ages.shape)
        nu = pm.Laplace("nu", mu=0, b=delta / epsilon)
        dp_mean = pm.Deterministic("dp_mean", mean + nu)

    return model


ages = pv.Uniform("ages", lower=0, upper=100, num_elements=20)
ds = pv.Dataset(input_specs=[ages])
program = pv.Program("output", dataset=ds, output_type=pv.Float, function=dp_mean)

trace: az.InferenceData = pv.infer(
    program, cores=4, draws=10_000, method=pv.Method.PYMC, use_new_method=True
)

print(trace["posterior"])
