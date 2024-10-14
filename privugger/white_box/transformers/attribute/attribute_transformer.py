from ...ast_transformer import AstTransformer
from .attribute_model import Attribute, AttributeOperation
import pytensor.tensor as pt
import pymc as pm
import ast


class AttributeTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Attribute):
        operand = super().to_custom_model(node.value)
        attribute = self.__map_attribute(node.attr)
        return Attribute(node.lineno, operand, attribute)

    def __map_attribute(self, attribute_name):
        if attribute_name == "size" or attribute_name == "len":
            return AttributeOperation.LEN

        if attribute_name == "sum":
            return AttributeOperation.SUM

        if attribute_name == "min":
            return AttributeOperation.MIN

        if attribute_name == "max":
            return AttributeOperation.MAX

        # Non 'common' attribute: Return the name of the attribute
        return attribute_name

    def to_pymc(self, node: Attribute, condition, in_function):
        (operand, size) = super().to_pymc(node.operand, condition, in_function)
        print(type(operand))
        if node.attribute == AttributeOperation.LEN:
            if size:
                return size
            return (
                operand.shape[0]
                if isinstance(operand, pt.TensorVariable)
                else len(operand)
            )

        if node.attribute == AttributeOperation.SUM:
            return (
                pm.math.sum(operand)
                if isinstance(operand, pt.TensorVariable)
                else sum(operand)
            )

        if node.attribute == AttributeOperation.MIN:
            return (
                pt.min(operand)
                if isinstance(operand, pt.TensorVariable)
                else sum(operand)
            )

        if node.attribute == AttributeOperation.MAX:
            return (
                pt.max(operand)
                if isinstance(operand, pt.TensorVariable)
                else sum(operand)
            )

        # TODO: Handle attributes that are not 'sum' and 'size'
        raise TypeError("Unsupported attribute")
