import privugger as pv
import arviz as az

def masking(ages):
    output = ages
    for i in range(len(ages)):
        if ages[i] >= 0 and ages[i] < 25:
            output[i] = 0.0
            if ages[i] < 20:
                output[i] = 0.5
                
        if ages[i] >= 25 and ages[i] < 50:
            output[i] = 1.0
            if ages[i] < 40:
                output[i] = 1.5
                
        if ages[i] >= 50 and ages[i] < 75:
            output[i] = 2.0
            if ages[i] < 60:
                output[i] = 2.5
        
        if ages[i] >= 75:
            output[i] = 3.0
            if ages[i] < 85:
                output[i] = 3.5

    return output

input = pv.DiscreteUniform("input", lower=0, upper=1, num_elements=100)
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
