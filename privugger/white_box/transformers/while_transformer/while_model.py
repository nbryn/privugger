from ...custom_node import CustomNode
from typing import List

class While(CustomNode):
    body: List[CustomNode] = []
    test: CustomNode = None
    
    def __init__(self, line_number, test, body):
        super().__init__("While", line_number)
        self.test = test
        self.body = body