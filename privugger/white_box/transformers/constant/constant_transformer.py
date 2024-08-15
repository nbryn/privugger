from ..constant.constant_model import Constant
from .. import AstTransformer
import ast


class ConstantTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Constant):
        return Constant(node.lineno, node if isinstance(node, int) else node.value)
