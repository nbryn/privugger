from ..operation.operation_model import Operation
from .. import CustomNode


# TODO: Compare doesn't have to be a operation, can also be function call
class Compare(CustomNode):
    operation: Operation = None
    right = None
    left = None

    def __init__(self, line_number, left, right, operation):
        super().__init__("Compare", line_number)
        self.left = left
        self.right = right
        self.operation = operation


# TODO: Compare2 doesn't have to be a operation, can also be function call
class Compare2(CustomNode):
    right_operation: Operation = None
    left_operation: Operation = None
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
