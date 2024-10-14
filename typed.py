import theano
import aesara.tensor as tt
import numpy as np


def method(ages):

    @theano.compile.ops.as_op(itypes=[tt.dvector], otypes=[tt.dscalar])
    def masking(ages):
        output = ages
        for i in range(len(ages)):
            if 0 <= ages[i] < 25:
                output[i] = 0
            if 25 <= ages[i] < 50:
                output[i] = 1
            if 50 <= ages[i] < 75:
                output[i] = 2
            if 75 <= ages[i]:
                output[i] = 3
        return np.array(output)
    return masking(ages)
