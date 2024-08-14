from .. import CustomNode, Reference


# Represents 'object.attribute'
# Attribute can be a 'normal' operation (like sum) or a custom attribute
# If its a operation it will be of type operation otherwise it will be a string
class Attribute(CustomNode):
    operand: Reference = None
    attribute = None

    def __init__(self, line_number, operand, attribute):
        super().__init__("Attribute", line_number)
        self.attribute = attribute
        self.operand = operand
