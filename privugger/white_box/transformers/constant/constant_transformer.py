from ...white_box_ast_transformer import WhiteBoxAstTransformer
from ..constant.constant_model import Constant
import ast


class ConstantTransformer(WhiteBoxAstTransformer):
    def to_custom_model(self, node: ast.Constant):
        return Constant(node.lineno, node if isinstance(node, int) else node.value)

    def to_pymc(self, node: Constant, _, __):
        return node.value
        