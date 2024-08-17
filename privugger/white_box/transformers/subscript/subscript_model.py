from ...custom_node import CustomNode


class Subscript(CustomNode):
    operand = ""
    lower = None
    upper = None

    def __init__(self, line_number, operand, lower, upper):
        super().__init__("Subscript", line_number)
        self.operand = operand
        self.lower = lower
        self.upper = upper
