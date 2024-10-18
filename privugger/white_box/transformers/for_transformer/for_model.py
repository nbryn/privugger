from ...custom_node import CustomNode
from typing import List

class For(CustomNode):
    body: List[CustomNode] = []
    loop_var: str
    start: CustomNode
    stop: CustomNode
    

    def __init__(self, line_number, loop_var, start, stop, body):
        super().__init__("Loop", line_number)
        self.loop_var = loop_var
        self.start = start
        self.stop = stop
        self.body = body
