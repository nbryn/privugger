from .. import AstTransformer
from .binop_model import BinOp
import ast


class BinOpTransformer(AstTransformer):
    def to_custom_model(self, node: ast.BinOp):
        operation = self.map_operation(node.op)
        right = self.map_to_custom_type(node.right)
        left = self.map_to_custom_type(node.left)

        return BinOp(node.lineno, left, right, operation)
