from ...custom_node import SastNode
from enum import Enum

class UnaryOperation(Enum):
    ADD = 1
    SUB = 2
    NOT = 3

class UnaryOp(SastNode):
    operation: UnaryOperation = None
    operand: SastNode = None

    def __init__(self, line_number, operand, operation):
        super().__init__("UnaryOp", line_number)
        self.operation = operation
        self.operand = operand