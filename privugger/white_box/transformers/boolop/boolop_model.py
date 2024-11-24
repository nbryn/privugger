from ...custom_node import SastNode
from enum import Enum


class BoolOperation(Enum):
    AND = 1
    OR = 2


class BoolOp(SastNode):
    operation: BoolOperation
    right: SastNode = None
    left: SastNode = None

    def __init__(self, line_number, left, right, operation):
        super().__init__("BoolOp", line_number)
        self.operation = operation
        self.right = right
        self.left = left
