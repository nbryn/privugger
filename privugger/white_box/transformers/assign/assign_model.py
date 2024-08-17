from ...custom_node import CustomNode


class Assign(CustomNode):
    value = None

    def __init__(self, variable_name, line_number, value):
        super().__init__(variable_name, line_number)
        self.value = value


# Represents assignment like: 'var[i] = 5'
class AssignIndex(Assign):
    index = None

    def __init__(self, variable_name, line_number, value, index):
        super().__init__(variable_name, line_number, value)
        self.index = index
