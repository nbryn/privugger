from ..operation.operation_model import Operation
from .attribute_model import Attribute
from .. import AstTransformer
import ast


class AttributeTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Attribute):
        operand = self.map_to_custom_type(node.value)
        return Attribute(node.lineno, operand, self.__map_attribute(node.attr))

    def __map_attribute(self, attribute_name):
        if attribute_name == "sum":
            return Operation.SUM

        if attribute_name == "size" or attribute_name == "len":
            return Operation.SIZE

        # Non 'common' attribute: Return the name of the attribute
        return attribute_name
