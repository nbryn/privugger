from .. import AstTransformer
from .binop_model import BinOp
import ast


class BinOpTransformer(AstTransformer):
    def to_custom_model(self, node: ast.BinOp):
        operation = self._map_operation(node.op)
        right = self._map_to_custom_type(node.right)
        left = self._map_to_custom_type(node.left)

        return BinOp(node.lineno, left, right, operation)

    def to_pymc(self, node: BinOp, pymc_model_builder, condition, in_function):
        left = pymc_model_builder.to_pymc(node.left, condition, in_function)
        right = pymc_model_builder.to_pymc(node.right, condition, in_function)
        if isinstance(left, tuple):
            left = left[0]

        if isinstance(right, tuple):
            right = right[0]

        return pymc_model_builder.to_pymc_operation(node.operation, left, right)