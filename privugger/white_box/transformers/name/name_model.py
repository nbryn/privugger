from ...custom_node import CustomNode


class Name(CustomNode):
    reference_to = ""

    def __init__(self, line_number, reference_to):
        super().__init__("Reference", line_number)
        self.reference_to = reference_to
