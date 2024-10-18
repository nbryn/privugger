from ...custom_node import CustomNode
from typing import List
from enum import Enum


class NumpyOperation(Enum):
    ARRAY = 1
    EXP = 2
    DOT = 3


class Numpy(CustomNode):
    def __init__(self, line_number, name):
        super().__init__(f"Numpy-{name}", line_number)


class NumpyFunction(Numpy):
    operation: NumpyOperation = None
    arguments: List[CustomNode] = []

    def __init__(self, line_number, operation: NumpyOperation, arguments):
        super().__init__(line_number, operation.name)
        self.operation = operation
        self.arguments = arguments


class NumpyExponential(Numpy):
    scale: CustomNode

    def __init__(self, line_number, scale):
        super().__init__(line_number, "Exponential")
        self.scale = scale


class NumpyPoisson(Numpy):
    lam: CustomNode

    def __init__(self, line_number, lam):
        super().__init__(line_number, "Poisson")
        self.lam = lam


class NumpyBinomial(Numpy):
    n: CustomNode
    p: CustomNode

    def __init__(self, line_number, n, p):
        super().__init__(line_number, "Binomial")
        self.n = n
        self.p = p


class NumpyUniform(Numpy):
    low: CustomNode
    high: CustomNode

    def __init__(self, line_number, low, high):
        super().__init__(line_number, "Uniform")
        self.high = high
        self.low = low


class NumpyNormal(Numpy):
    scale: CustomNode
    loc: CustomNode

    def __init__(self, line_number, loc, scale):
        super().__init__(line_number, "Normal")
        self.scale = scale
        self.loc = loc


class NumpyLaplace(Numpy):
    scale: CustomNode
    loc: CustomNode

    def __init__(self, line_number, loc, scale):
        super().__init__(line_number, "Laplace")
        self.scale = scale
        self.loc = loc
