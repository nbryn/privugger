from ...custom_node import SastNode
from typing import List

class Call(SastNode):
    arguments: List[SastNode] = []
    operand = None

    def __init__(self, line_number, operand, arguments):
        super().__init__("Call", line_number)
        self.arguments = arguments
        self.operand = operand