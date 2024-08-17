from ...custom_node import CustomNode
from typing import List

class For(CustomNode):
    body: List[CustomNode] = []
    start = None
    stop = None

    def __init__(self, line_number, start, stop, body):
        super().__init__("Loop", line_number)
        self.start = start
        self.stop = stop
        self.body = body
