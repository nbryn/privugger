from ...white_box_ast_transformer import WhiteBoxAstTransformer
from .subscript_model import Subscript
from ..index.index_model import Index
import ast


class SubscriptTransformer(WhiteBoxAstTransformer):
    def to_custom_model(self, node: ast.Subscript):
        if isinstance(node.slice, ast.Index):
            index = (
                node.slice.value.value
                if hasattr(node.slice.value, "value")
                else node.slice.value.id
            )
            return Index(node.lineno, node.value.id, index)

        lower = super().to_custom_model(node.slice.lower)
        upper = super().to_custom_model(node.slice.upper)
        dependency_name = node.value.id

        return Subscript(node.lineno, dependency_name, lower, upper)

    def to_pymc(self, node: Subscript, condition, in_function):
        (operand, size) = self.program_variables[node.operand]
        lower = super().to_pymc(node.lower, condition, in_function) if node.lower else 0
        upper = (
            super().to_pymc(node.upper, condition, in_function) if node.upper else size
        )

        return operand[lower:upper]
