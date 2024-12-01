import privugger as pv
import arviz as az

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

input = pv.DiscreteUniform("input", lower=0, upper=1, num_elements=20)
dataset = pv.Dataset(input_specs=[input])


program = pv.Program(
    name="output",
    dataset=dataset,
    output_type=pv.Float,
    function=masking,
)

trace: az.InferenceData = pv.infer(
    program, cores=4, draws=10_000, method=pv.Method.PYMC, use_new_method=True
)

print(trace["posterior"])
