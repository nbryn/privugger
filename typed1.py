import theano
import theano.tensor as tt
import numpy as np


def method(ages):

    @theano.compile.ops.as_op(itypes=[tt.dvector], otypes=[tt.dscalar])
    def analyze_works(ages):
        subset = ages[:20]
        avg = subset.sum() / subset.size
        return np.array(avg)
    return analyze_works(ages)
