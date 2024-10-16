from ...ast_transformer import AstTransformer
from .list_model import List
import ast


class ListTransformer(AstTransformer):
    def to_custom_model(self, node: ast.List):
        values = list(map(super().to_custom_model, node.elts))
        return List(node.lineno, values)

    def to_pymc(self, node: List, _, __):
        return list(map(super().to_pymc, node.values))
