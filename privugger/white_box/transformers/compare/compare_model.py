from ...custom_node import CustomNode
from enum import Enum

class CompareOperation(Enum):
    EQUAL = 1
    LT = 2
    LTE = 3
    GT = 4
    GTE = 5

# TODO: Compare doesn't have to be a operation, can also be function call
class Compare(CustomNode):
    operation: CompareOperation = None
    right = None
    left = None

    def __init__(self, line_number, left, right, operation):
        super().__init__("Compare", line_number)
        self.left = left
        self.right = right
        self.operation = operation


# TODO: Compare2 doesn't have to be a operation, can also be function call
class Compare2(CustomNode):
    right_operation: CompareOperation = None
    left_operation: CompareOperation = None
    middle = None
    right = None
    left = None

    def __init__(
        self, line_number, left, left_operation, middle, right, right_operation
    ):
        super().__init__("Compare2", line_number)
        self.right_operation = right_operation
        self.left_operation = left_operation
        self.middle = middle
        self.right = right
        self.left = left



