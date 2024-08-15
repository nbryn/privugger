from ..operation.operation_model import Operation
from .attribute_model import Attribute
from .. import AstTransformer
import ast


class AttributeTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Attribute):
        operand = self._map_to_custom_type(node.value)
        return Attribute(node.lineno, operand, self.__map_attribute(node.attr))

    def __map_attribute(self, attribute_name):
        if attribute_name == "sum":
            return Operation.SUM

        if attribute_name == "size" or attribute_name == "len":
            return Operation.SIZE

        # Non 'common' attribute: Return the name of the attribute
        return attribute_name
    
    def to_pymc(self, node: Attribute, pymc_model_builder, condition, in_function):
        (operand, size) = pymc_model_builder.to_pymc(node.operand, condition, in_function)
        if node.attribute == Operation.SIZE:
            return size

        # TODO: Handle attributes that are not 'sum' and 'size'
        return pymc_model_builder.to_pymc_operation(node.attribute, operand)
