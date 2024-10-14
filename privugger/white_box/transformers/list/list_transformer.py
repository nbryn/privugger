from ...ast_transformer import AstTransformer
from .list_model import ListNode
import ast


class ListTransformer(AstTransformer):
    def to_custom_node(self, node: ast.List):
        values = list(map(super().to_custom_model, node.elts))
        return ListNode(node.lineno, values)

    def to_pymc(self, node: ListNode, _, __):
        return list(map(super().to_pymc, node.values))
