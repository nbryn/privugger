from ...custom_node import SastNode
from enum import Enum


class ArithmeticOperation(Enum):
    ADD = 1
    SUB = 2
    DIVIDE = 3
    MULTIPLY = 4


class BinOp(SastNode):
    operation: ArithmeticOperation = None
    right: SastNode = None
    left: SastNode = None

    def __init__(self, line_number, left, right, operation):
        super().__init__("BinOp", line_number)
        self.operation = operation
        self.right = right
        self.left = left
