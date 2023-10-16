import theano
import theano.tensor as tt
import numpy as np


def method(ages):

    @theano.compile.ops.as_op(itypes=[tt.dvector], otypes=[tt.dscalar])
    def analyze_works(ages):
        total = ages.sum()
        return np.array(total)
    return analyze_works(ages)
