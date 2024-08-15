from .list_model import ListNode
from .. import AstTransformer
import ast


class ListTransformer(AstTransformer):
    def __init__(self, ast_parser):
        super().__init__(ast_parser)

    def to_custom_node(self, node: ast.List):
        values = list(map(self._map_to_custom_type, node.elts))
        return ListNode(node.lineno, values)
