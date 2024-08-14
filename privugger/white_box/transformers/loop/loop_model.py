from .. import CustomNode
from typing import List

class Loop(CustomNode):
    body: List[CustomNode] = []
    start = None
    stop = None

    def __init__(self, line_number, start, stop, body):
        super().__init__("Loop", line_number)
        self.start = start
        self.stop = stop
        self.body = body
