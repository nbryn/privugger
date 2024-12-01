from ... import custom_node


class Index(custom_node.SastNode):
    operand = ""
    first_index = None
    second_index = None

    def __init__(self, line_number, operand, first_index, second_index=None):
        super().__init__("Index", line_number)
        self.operand = operand
        self.first_index = first_index
        self.second_index = second_index
