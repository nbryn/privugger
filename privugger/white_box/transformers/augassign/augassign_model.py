from ...custom_node import CustomNode
from ..name.name_model import Name
from enum import Enum


class AugAssignOperation(Enum):
    ADD = 1
    SUB = 2


class AugAssign(CustomNode):
    operation: AugAssignOperation
    value: CustomNode = None
    operand = Name

    def __init__(self, line_number, operand, operation, value):
        super().__init__("AugAssign", line_number)
        self.operation = operation
        self.operand = operand
        self.value = value
