from ..operation.operation_model import Operation
from ...custom_node import CustomNode

class UnaryOp(CustomNode):
    operation: Operation = None
    operand: CustomNode = None

    def __init__(self, line_number, operand, operation):
        super().__init__("UnaryOp", line_number)
        self.operation = operation
        self.operand = operand