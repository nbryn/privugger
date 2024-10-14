from ...ast_transformer import AstTransformer
from .boolop_model import BoolOp, BoolOperation
import pymc.math as pm_math
import numpy as np
import ast

class BoolOpTransformer(AstTransformer):
    def to_custom_model(self, node: ast.BoolOp):
        operation = self._to_custom_operation(node.op)
        right = super().to_custom_model(node.values[0])
        left = super().to_custom_model(node.values[1])
        
        return BoolOp(node.lineno, left, right, operation)
        
    
    def to_pymc(self, node: BoolOp, condition, in_function):
        right = super().to_pymc(node.right, condition, in_function)
        left = super().to_pymc(node.left, condition, in_function)
        
        if isinstance(left, tuple):
            left = left[0]

        if isinstance(right, tuple):
            right = right[0]

        return self._to_pymc_operation(node.operation, left, right)
    
    
    def _to_custom_operation(self, operation: ast):
        if isinstance(operation, ast.And):
            return BoolOperation.AND

        if isinstance(operation, ast.Or):
            return BoolOperation.OR
        
        print(operation)
        raise TypeError("Unsupported BoolOp operation")
    
    def _to_pymc_operation(self, operation: BoolOperation, left, right):
        if operation == BoolOperation.AND:
            return pm_math.and_(np.array(left), np.array(right))
        
        if operation == BoolOperation.OR:
            return pm_math.or_(np.array(left), np.array(right))

        print(operation)
        raise TypeError("Unsupported BoolOp operation")
        