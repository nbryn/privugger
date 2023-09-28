import theano
import theano.tensor as tt
import numpy as np


def method(ages):

    @theano.compile.ops.as_op(itypes=[tt.dvector], otypes=[tt.dscalar])
    def avg_or_sum(ages):
        return np.array(ages.sum() / ages.size)
    return avg_or_sum(ages)
