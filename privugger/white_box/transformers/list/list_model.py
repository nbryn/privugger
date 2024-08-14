from .. import CustomNode


class ListNode(CustomNode):
    values = []

    def __init__(self, line_number, values):
        super().__init__("List", line_number)
        self.values = values
