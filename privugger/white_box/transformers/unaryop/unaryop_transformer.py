from .unaryop_model import UnaryOp
from .. import AstTransformer
import ast


class UnaryOpTransformer(AstTransformer):
    def __init__(self, ast_parser):
        self.ast_parser = ast_parser

    def to_custom_model(self, node: ast.UnaryOp):
        operation = self._map_operation(node.op)
        operand = self._map_to_custom_type(node.operand)

        return UnaryOp(node.lineno, operand, operation)
