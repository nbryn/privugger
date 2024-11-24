from ...custom_node import SastNode

class Return(SastNode):
    value = None

    def __init__(self, line_number, value):
        super().__init__("return", line_number)
        self.value = value