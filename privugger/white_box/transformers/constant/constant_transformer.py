from ..constant.constant_model import Constant
from .. import AstTransformer
import ast


class ConstantTransformer(AstTransformer):
    def __init__(self, ast_parser):
        super().__init__(ast_parser)

    def to_custom_model(self, node: ast.Constant):
        return Constant(node.lineno, node if isinstance(node, int) else node.value)
