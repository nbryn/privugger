from ...custom_node import CustomNode
from enum import Enum

class UnaryOperation(Enum):
    ADD = 1
    SUB = 2

class UnaryOp(CustomNode):
    operation: UnaryOperation = None
    operand: CustomNode = None

    def __init__(self, line_number, operand, operation):
        super().__init__("UnaryOp", line_number)
        self.operation = operation
        self.operand = operand