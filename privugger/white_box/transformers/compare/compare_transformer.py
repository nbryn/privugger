from ...ast_transformer import AstTransformer
from .compare_model import Compare, Compare2, CompareOperation
import pymc.math as pm_math
from typing import Union
import ast


class CompareTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Compare):
        left = super().to_custom_model(node.left)
        # Assumes max two comparators
        left_operation = self._to_custom_operation(node.ops[0])
        middle_or_right = super().to_custom_model(node.comparators[0])
        if len(node.comparators) < 2:
            return Compare(node.lineno, left, middle_or_right, left_operation)

        right_operation = self._to_custom_operation(node.ops[1])
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
            return self._to_pymc_operation(node.operation, left, right)

        middle = super().to_pymc(node.middle, condition, in_function)
        left_compare = self._to_pymc_operation(node.left_operation, left, middle)
        right_compare = self._to_pymc_operation(node.right_operation, middle, right)

        return pm_math.and_(left_compare, right_compare)

    def _to_custom_operation(self, operation):
        if isinstance(operation, ast.Eq):
            return CompareOperation.EQUAL
        
        if isinstance(operation, ast.Lt):
            return CompareOperation.LT

        if isinstance(operation, ast.LtE):
            return CompareOperation.LTE

        if isinstance(operation, ast.Gt):
            return CompareOperation.GT

        if isinstance(operation, ast.GtE):
            return CompareOperation.GTE

        print(operation)
        raise TypeError("Unknown AST operation")
    
    def _to_pymc_operation(self, operation: CompareOperation, operand, right=None):
        if operation == CompareOperation.EQUAL:
            return pm_math.eq(operand, right)

        if operation == CompareOperation.LT:
            return pm_math.lt(operand, right)

        if operation == CompareOperation.LTE:
            return pm_math.le(operand, right)

        if operation == CompareOperation.GT:
            return pm_math.gt(operand, right)

        if operation == CompareOperation.GTE:
            return pm_math.ge(operand, right)
        
        print(operation)
        raise TypeError("Unsupported operation")