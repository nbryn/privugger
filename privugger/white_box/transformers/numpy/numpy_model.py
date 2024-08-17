from ...custom_node import CustomNode
from typing import List
from enum import Enum
from abc import ABC

class NumpyOperation(Enum):
    ARRAY = 1
    EXP = 2
    DOT = 3


class NumpyDistributionType(Enum):
    NORMAL = "Normal"
    LAPLACE = "Laplace"


class Numpy(CustomNode):
    def __init__(self, line_number, name):
        super().__init__(f"Numpy {name}", line_number)

class NumpyFunction(Numpy):
    operation: NumpyOperation = None
    arguments: List[CustomNode] = []

    def __init__(self, line_number, operation: NumpyOperation, arguments):
        super().__init__(line_number, operation.value, arguments)
        self.operation = operation
        self.arguments = arguments


class NumpyDistribution(Numpy):
    distribution: NumpyDistributionType
    scale = None
    loc = None

    def __init__(self, line_number, distribution: NumpyDistributionType, scale, loc):
        super().__init__(line_number, distribution.value)
        self.distribution = distribution
        self.scale = scale
        self.loc = loc
