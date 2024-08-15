from .. import CustomNode
from typing import List


class FunctionDef(CustomNode):
    arguments: List[CustomNode] = []
    body: List[CustomNode] = []

    def __init__(self, name, line_number, arguments, body):
        super().__init__(name, line_number)
        self.arguments = arguments
        self.body = body
