from ..constant.constant_model import Constant
from .. import AstTransformer
import ast


class ConstantTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Constant):
        return Constant(node.lineno, node if isinstance(node, int) else node.value)

    def to_pymc(self, node: Constant, pymc_model_builder, condition, in_function):
        return node.value
        