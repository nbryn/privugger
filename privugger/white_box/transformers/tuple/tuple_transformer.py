from ...ast_transformer import AstTransformer
from .tuple_model import Tuple
import ast


class TupleTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Tuple):
        args = tuple(map(super().to_custom_model, node.elts))
        return Tuple(node.lineno, args)

    def to_pymc(self, node: Tuple, _, __):
        return tuple(map(super().to_pymc, node.tuple))
