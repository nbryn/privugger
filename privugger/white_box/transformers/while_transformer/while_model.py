from ...custom_node import SastNode
from typing import List

class While(SastNode):
    body: List[SastNode] = []
    test: SastNode = None
    
    def __init__(self, line_number, test, body):
        super().__init__("While", line_number)
        self.test = test
        self.body = body