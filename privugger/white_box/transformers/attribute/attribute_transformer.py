from ...ast_transformer import AstTransformer
from .attribute_model import Attribute, AttributeOperation
import pytensor.tensor as pt
import pymc as pm
import ast


class AttributeTransformer(AstTransformer):
    def to_sast(self, node: ast.Attribute):
        operand = super().to_sast(node.value)
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
        
        if attribute_name == "abs":
            return AttributeOperation.ABS

        # Attribute not related to Python library function: Return the name of the attribute
        return attribute_name

    def to_pymc(self, node: Attribute, conditions: dict, in_function):
        operand = super().to_pymc(node.operand, conditions, in_function)
        size = None
        if isinstance(operand, tuple):
            size = operand[1]
            operand = operand[0]

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
                else min(operand)
            )

        if node.attribute == AttributeOperation.MAX:
            return (
                pt.max(operand)
                if isinstance(operand, pt.TensorVariable)
                else max(operand)
            )
            
        if node.attribute == AttributeOperation.ABS:
            return (
                pm.math.abs(operand)
                if isinstance(operand, pt.TensorVariable)
                else abs(operand)
            )

        raise TypeError("Unsupported attribute")
