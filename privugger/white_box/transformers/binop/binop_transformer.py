from ...white_box_ast_transformer import WhiteBoxAstTransformer
from .binop_model import BinOp
import ast


class BinOpTransformer(WhiteBoxAstTransformer):
    def to_custom_model(self, node: ast.BinOp):
        operation = super()._map_operation(node.op)
        right = super().to_custom_model(node.right)
        left = super().to_custom_model(node.left)

        return BinOp(node.lineno, left, right, operation)

    def to_pymc(self, node: BinOp, condition, in_function):
        left = super().to_pymc(node.left, condition, in_function)
        right = super().to_pymc(node.right, condition, in_function)
        if isinstance(left, tuple):
            left = left[0]

        if isinstance(right, tuple):
            right = right[0]

        return super()._to_pymc_operation(node.operation, left, right)