import theano
import theano.tensor as tt
import numpy as np


def method(ages):

    @theano.compile.ops.as_op(itypes=[tt.dvector], otypes=[tt.dscalar])
    def analyze_works(ages):
        first = ages[0]
        return np.array(first)
    return analyze_works(ages)
