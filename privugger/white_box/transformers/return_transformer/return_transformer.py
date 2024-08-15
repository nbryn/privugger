from .return_model import Return
from .. import AstTransformer
import ast


class ReturnTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Return):
        value = self._map_to_custom_type(node.value)
        return Return(node.lineno, value)
