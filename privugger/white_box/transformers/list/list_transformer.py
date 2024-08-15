from .list_model import ListNode
from .. import AstTransformer
import ast


class ListTransformer(AstTransformer):
    def to_custom_node(self, node: ast.List):
        values = list(map(self._map_to_custom_type, node.elts))
        return ListNode(node.lineno, values)
    
    def to_pymc(self, node: ListNode, pymc_model_builder):
        return list(map(pymc_model_builder.to_pymc, node.values))
