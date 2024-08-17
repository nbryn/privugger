from ...white_box_ast_transformer import WhiteBoxAstTransformer
from .compare_model import Compare, Compare2
import pymc3.math as pm_math
from typing import Union
import ast


class CompareTransformer(WhiteBoxAstTransformer):
    def to_custom_model(self, node: ast.Compare):
        left = super().to_custom_model(node.left)
        # Assumes max two comparators
        left_operation = super()._map_operation(node.ops[0])
        middle_or_right = super().to_custom_model(node.comparators[0])
        if len(node.comparators) < 2:
            return Compare(node.lineno, left, middle_or_right, left_operation)

        right_operation = super()._map_operation(node.ops[1])
        right = super().to_custom_model(node.comparators[1])

        return Compare2(
            node.lineno, left, left_operation, middle_or_right, right, right_operation
        )

        
    def to_pymc(self, node: Union[Compare, Compare2], condition, in_function):
        left = super().to_pymc(node.left, condition, in_function)
        right = super().to_pymc(node.right, condition, in_function)

        if isinstance(left, tuple):
            left = left[0]

        if isinstance(node, Compare):
            return super()._to_pymc_operation(node.operation, left, right)

        middle = super().to_pymc(node.middle, condition, in_function)
        left_compare = super()._to_pymc_operation(node.left_operation, left, middle)
        right_compare = super()._to_pymc_operation(node.right_operation, middle, right)

        return pm_math.and_(left_compare, right_compare)

        