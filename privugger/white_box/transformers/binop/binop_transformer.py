from ...ast_transformer import AstTransformer
from .binop_model import BinOp, ArithmeticOperation
import ast


class BinOpTransformer(AstTransformer):
    def to_custom_model(self, node: ast.BinOp):
        operation = self._to_custom_operation(node.op)
        right = super().to_custom_model(node.right)
        left = super().to_custom_model(node.left)

        return BinOp(node.lineno, left, right, operation)

    def to_pymc(self, node: BinOp, condition, in_function):
        right = super().to_pymc(node.right, condition, in_function)
        left = super().to_pymc(node.left, condition, in_function)
        
        if isinstance(left, tuple):
            left = left[0]

        if isinstance(right, tuple):
            right = right[0]

        return self._to_pymc_operation(node.operation, left, right)

    def _to_custom_operation(self, operation: ast):
        if isinstance(operation, ast.Add):
            return ArithmeticOperation.ADD

        if isinstance(operation, ast.Sub):
            return ArithmeticOperation.SUB

        if isinstance(operation, ast.Div):
            return ArithmeticOperation.DIVIDE

        if isinstance(operation, ast.Mult):
            return ArithmeticOperation.MULTIPLY

        print(operation)
        raise TypeError("Unknown arithmetic operation")

    def _to_pymc_operation(self, operation: ArithmeticOperation, left, right):
        if operation == ArithmeticOperation.DIVIDE:
            return left / right

        if operation == ArithmeticOperation.SUB:
            return left - right

        if operation == ArithmeticOperation.ADD:
            return left + right

        if operation == ArithmeticOperation.MULTIPLY:
            return left * right

        print(operation)
        raise TypeError("Unsupported operation")
