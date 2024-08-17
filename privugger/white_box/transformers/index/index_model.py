from ... import custom_node


class Index(custom_node.CustomNode):
    operand = ""
    index = None

    def __init__(self, line_number, operand, index):
        super().__init__("Index", line_number)
        self.operand = operand
        self.index = index
