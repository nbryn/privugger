from ...ast_transformer import AstTransformer
from .tuple_model import Tuple
import ast


class TupleTransformer(AstTransformer):
    def to_sast(self, node: ast.Tuple):
        args = tuple(map(super().to_sast, node.elts))
        return Tuple(node.lineno, args)

    def to_pymc(self, node: Tuple, _, __):
        return tuple(map(super().to_pymc, node.tuple))
