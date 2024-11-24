from ...custom_node import SastNode
from typing import List


class If(SastNode):
    orelse: List[SastNode] = []
    body: List[SastNode] = []
    parent_if: SastNode = None
    has_break_in_body = False
    condition = None

    def __init__(self, line_number, condition, body, orelse, has_break_in_body):
        super().__init__("if", line_number)
        self.has_break_in_body = has_break_in_body
        self.condition = condition
        self.orelse = orelse
        self.body = body
