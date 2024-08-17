from ...custom_node import CustomNode

class Return(CustomNode):
    value = None

    def __init__(self, line_number, value):
        super().__init__("return", line_number)
        self.value = value