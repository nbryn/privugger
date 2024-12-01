from ...custom_node import SastNode
from typing import Tuple


class Tuple(SastNode):
    tuple = Tuple[SastNode]

    def __init__(self, line_number, tuple):
        super().__init__("Tuple", line_number)
        self.tuple = tuple
