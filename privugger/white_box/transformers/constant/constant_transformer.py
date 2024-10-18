from ...ast_transformer import AstTransformer
from ..constant.constant_model import Constant
import ast


class ConstantTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Constant):
        return Constant(node.lineno, node if isinstance(node, int) else node.value)

    def to_pymc(self, node: Constant, _, __):
        return node.value
