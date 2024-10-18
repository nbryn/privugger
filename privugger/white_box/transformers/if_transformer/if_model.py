from ...custom_node import CustomNode
from typing import List

class If(CustomNode):
    orelse: List[CustomNode] = []
    body: List[CustomNode] = []
    has_break_in_body = False
    condition = None
    
    def __init__(self, line_number, condition, body, orelse, has_break_in_body):
        super().__init__("if", line_number)
        self.has_break_in_body = has_break_in_body
        self.condition = condition
        self.orelse = orelse 
        self.body = body