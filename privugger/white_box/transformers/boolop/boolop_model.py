from ...custom_node import CustomNode
from enum import Enum


class BoolOperation(Enum):
    AND = 1
    OR = 2


class BoolOp(CustomNode):
    operation: BoolOperation
    right: CustomNode = None
    left: CustomNode = None

    def __init__(self, line_number, left, right, operation):
        super().__init__("BoolOp", line_number)
        self.operation = operation
        self.right = right
        self.left = left
