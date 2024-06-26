# White-box analysis:
My code can be found in privugger/white_box

# The following demonstrates how to use my method like described in my paper:

def ages_dp(ages):
    avg = ages.sum() / ages.size
    epsilon = 0.1
    delta = 100 / ages.size # assumes ages are in the interval [0-100]
    nu = np.random.laplace(loc=0.0, scale=delta / epsilon)
    dp_avg = avg + nu

    return dp_avg

ages = pv.Uniform("ages", lower=0, upper=100, num_elements=20)
ds = pv.Dataset(input_specs=[ages])
program = pv.Program("output", dataset=ds, output_type=pv.Float, function=ages_dp)

trace: az.InferenceData = pv.infer(program, cores=4, draws=10_000, method="pymc3", use_new_method=True)

mi_avg = pv.mi_sklearn(trace, var_names=["ages0 - 2", "avg - 3"])
mi_dp_avg = pv.mi_sklearn(trace, var_names=["ages0 - 2", "dp_avg - 7"])

print(mi_avg[0])
print(mi_dp_avg[0])

# Privugger: Data Privacy Debugger

Docs and tutorials: https://itu-square.github.io/privugger/

Privugger (/prɪvʌɡə(r)/) is a privacy risk analysis library for python
programs.  Privugger, takes as input a python program and a
specification of the adversary's knowledge about the input of the
program (the _prior knowledge_), and it returns a wide variety of
privacy risk analyses, including the following leakage measures:

* Knowledge-based probability queries
* Entropy
* Mutual Information
* KL-divergence
* min-entropy
* Bayes risk
* ...

Furthermore, Privugger is equipped with a module to perform _automatic
attacker synthesis_. That is, given a program and a leakage measure,
it finds the adversary's prior knowledge that maximizes the
leakage. In other words, it tells us what is the minimum amount of
information that the adversary must know in order for the program to
exhibit privacy risks. If this knowledge is publicly available, then
the program does not effectively protect users' privacy.



## Installation 

Privugger is a tool written entirely in Python and can be installed using the pip packet manager.

`pip install privugger`

## Usage

See the [docs and tutorials](https://itu-square.github.io/privugger/) for getting started with privugger!

