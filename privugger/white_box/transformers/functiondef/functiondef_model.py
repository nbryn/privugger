from ...custom_node import SastNode
from typing import List


class FunctionDef(SastNode):
    arguments: List[SastNode] = []
    body: List[SastNode] = []

    def __init__(self, name, line_number, arguments, body):
        super().__init__(name, line_number)
        self.arguments = arguments
        self.body = body
