from ...custom_node import CustomNode
from typing import List

class If(CustomNode):
    orelse: List[CustomNode] = []
    body: List[CustomNode] = []
    condition = None

    def __init__(self, line_number, condition, body, orelse):
        super().__init__("if", line_number)
        self.condition = condition
        self.orelse = orelse 
        self.body = body