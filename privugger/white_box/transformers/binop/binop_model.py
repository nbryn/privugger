from ..operation.operation_model import Operation
from .. import CustomNode

class BinOp(CustomNode):
    operation: Operation = None
    right: CustomNode = None
    left: CustomNode = None

    def __init__(self, line_number, left, right, operation):
        super().__init__("BinOp", line_number)
        self.operation = operation
        self.right = right
        self.left = left