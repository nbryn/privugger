from .subscript_model import Subscript
from ..index.index_model import Index
from .. import AstTransformer
import ast


class SubscriptTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Subscript):
        if isinstance(node.slice, ast.Index):
            index = (
                node.slice.value.value
                if hasattr(node.slice.value, "value")
                else node.slice.value.id
            )
            return Index(node.lineno, node.value.id, index)

        lower = self.map_to_custom_type(node.slice.lower)
        upper = self.map_to_custom_type(node.slice.upper)
        dependency_name = node.value.id

        return Subscript(node.lineno, dependency_name, lower, upper)
