from .. import CustomNode
from typing import List

class Call(CustomNode):
    arguments: List[CustomNode] = []
    operand = None

    def __init__(self, line_number, operand, arguments):
        super().__init__("Call", line_number)
        self.arguments = arguments
        self.operand = operand