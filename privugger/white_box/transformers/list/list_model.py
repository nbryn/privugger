from ...custom_node import SastNode


class List(SastNode):
    values = []

    def __init__(self, line_number, values):
        super().__init__("List", line_number)
        self.values = values
