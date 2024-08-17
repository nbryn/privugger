from ...white_box_ast_transformer import WhiteBoxAstTransformer
from ..operation.operation_model import Operation
from .attribute_model import Attribute
import ast


class AttributeTransformer(WhiteBoxAstTransformer):
    def to_custom_model(self, node: ast.Attribute):
        operand = super().to_custom_model(node.value)
        return Attribute(node.lineno, operand, self.__map_attribute(node.attr))

    def __map_attribute(self, attribute_name):
        if attribute_name == "sum":
            return Operation.SUM

        if attribute_name == "size" or attribute_name == "len":
            return Operation.SIZE

        # Non 'common' attribute: Return the name of the attribute
        return attribute_name

    def to_pymc(self, node: Attribute, condition, in_function):
        (operand, size) = super().to_pymc(node.operand, condition, in_function)
        if node.attribute == Operation.SIZE:
            return size

        # TODO: Handle attributes that are not 'sum' and 'size'
        return super()._to_pymc_operation(node.attribute, operand)
