from .compare_model import Compare, Compare2
from .. import AstTransformer
import pymc3.math as pm_math
from typing import Union
import ast


class CompareTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Compare):
        left = self._map_to_custom_type(node.left)
        # Assumes max two comparators
        left_operation = self._map_operation(node.ops[0])
        middle_or_right = self._map_to_custom_type(node.comparators[0])
        if len(node.comparators) < 2:
            return Compare(node.lineno, left, middle_or_right, left_operation)

        right_operation = self._map_operation(node.ops[1])
        right = self._map_to_custom_type(node.comparators[1])

        return Compare2(
            node.lineno, left, left_operation, middle_or_right, right, right_operation
        )

        
    def to_pymc(self, node: Union[Compare, Compare2], pymc_model_builder, condition, in_function):
        left = pymc_model_builder.to_pymc(node.left, condition, in_function)
        right = pymc_model_builder.to_pymc(node.right, condition, in_function)

        if isinstance(left, tuple):
            left = left[0]

        if isinstance(node, Compare):
            return pymc_model_builder.to_pymc_operation(node.operation, left, right)

        middle = pymc_model_builder.to_pymc(node.middle, condition, in_function)
        left_compare = pymc_model_builder.to_pymc_operation(node.left_operation, left, middle)
        right_compare = pymc_model_builder.to_pymc_operation(node.right_operation, middle, right)

        return pm_math.and_(left_compare, right_compare)

        