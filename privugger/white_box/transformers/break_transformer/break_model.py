from ...custom_node import SastNode


class Break(SastNode):
    def __init__(self, line_number):
        super().__init__("Break", line_number)
