from ...custom_node import CustomNode
from ..name.name_model import Name
from enum import Enum

class AttributeOperation(Enum):
    LEN = 1
    SUM = 2
    MIN = 3
    MAX = 4

# Represents 'object.attribute'
# Attribute can be a standard python function (like sum) or a custom attribute
# If it's a operation it will be of type operation otherwise it will be a string
class Attribute(CustomNode):
    operand: Name = None
    attribute = None

    def __init__(self, line_number, operand, attribute):
        super().__init__("Attribute", line_number)
        self.attribute = attribute
        self.operand = operand
