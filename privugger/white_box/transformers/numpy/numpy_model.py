import privugger.white_box.model as model
from typing import List
from enum import Enum

class Operation(Enum):
    ARRAY = 1
    EXP = 2
    DOT = 3

class DistributionType(Enum):
    NORMAL = "Normal"
    LAPLACE = "Laplace"

class Numpy(model.CustomNode):
    def __init__(self, line_number, name):
        model.CustomNode.__init__(self, f"Numpy {name}", line_number)


class Function(Numpy):
    operation: Operation = None
    arguments: List[model.CustomNode] = []

    def __init__(self, line_number, operation: Operation, arguments):
        Numpy.__init__(self, line_number, operation.value, arguments)
        self.operation = operation
        self.arguments = arguments


class Distribution(Numpy):
    distribution: DistributionType
    scale = None
    loc = None

    def __init__(self, line_number, distribution: DistributionType, scale, loc):
        Numpy.__init__(self, line_number, distribution.value)
        self.distribution = distribution
        self.scale = scale
        self.loc = loc