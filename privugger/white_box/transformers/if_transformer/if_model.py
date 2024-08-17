from ... import custom_node
from typing import List

class If(custom_node.CustomNode):
    body: List[custom_node.CustomNode] = []
    condition = None

    def __init__(self, line_number, condition, body):
        super().__init__("if", line_number)
        self.condition = condition
        self.body = body