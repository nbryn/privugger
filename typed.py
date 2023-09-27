import theano
import theano.tensor as tt
import numpy as np


def method(ages):

    @theano.compile.ops.as_op(itypes=[tt.dvector], otypes=[tt.dscalar])
    def avg(ages):
        return ages.sum()
        return np.array(ages.sum() / ages.size)
    return avg(ages)
