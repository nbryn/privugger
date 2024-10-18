from ...ast_transformer import AstTransformer
from .unaryop_model import UnaryOp, UnaryOperation
import pymc as pm
import ast


class UnaryOpTransformer(AstTransformer):
    def to_custom_model(self, node: ast.UnaryOp):
        operation = self._to_custom_operation(node.op)
        operand = super().to_custom_model(node.operand)

        return UnaryOp(node.lineno, operand, operation)

    def to_pymc(self, node: UnaryOp, condition, in_function):
        operand = super().to_pymc(node.operand, condition, in_function)
        if isinstance(operand, tuple):
            operand = operand[0]

        return self._to_pymc_operation(node.operation, operand)
    
    def _to_custom_operation(self, operation: ast):
        if isinstance(operation, ast.UAdd):
            return UnaryOperation.ADD

        if isinstance(operation, ast.USub):
            return UnaryOperation.SUB
        
        if isinstance(operation, ast.Not):
            return UnaryOperation.NOT

        print(operation)
        raise TypeError("Unknown UnaryOp operation")

    def _to_pymc_operation(self, operation: UnaryOperation, operand):
        if operation == UnaryOperation.SUB:
            return -operand

        if operation == UnaryOperation.ADD:
            return +operand
        
        if operation == UnaryOperation.NOT:
            return not operand

        print(operation)
        raise TypeError("Unsupported UnaryOp operation")