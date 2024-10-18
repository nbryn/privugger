from ...custom_node import CustomNode


class Break(CustomNode):
    def __init__(self, line_number):
        super().__init__("Break", line_number)
