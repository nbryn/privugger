from ...custom_node import SastNode
from ..name.name_model import Name
from enum import Enum


class AugAssignOperation(Enum):
    ADD = 1
    SUB = 2


class AugAssign(SastNode):
    operation: AugAssignOperation
    value: SastNode = None
    operand = Name

    def __init__(self, line_number, operand, operation, value):
        super().__init__("AugAssign", line_number)
        self.operation = operation
        self.operand = operand
        self.value = value
