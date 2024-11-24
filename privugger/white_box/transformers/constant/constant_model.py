from ...custom_node import SastNode


class Constant(SastNode):
    value = None

    def __init__(self, line_number, value):
        super().__init__("Constant", line_number)
        self.value = value
