from ...custom_node import SastNode
from typing import List

class For(SastNode):
    body: List[SastNode] = []
    loop_var: str
    start: SastNode
    stop: SastNode
    

    def __init__(self, line_number, loop_var, start, stop, body):
        super().__init__("Loop", line_number)
        self.loop_var = loop_var
        self.start = start
        self.stop = stop
        self.body = body
