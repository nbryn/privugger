from ...ast_transformer import AstTransformer
from ..index.index_transformer import IndexTransformer
from .subscript_model import Subscript
import ast


class SubscriptTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Subscript):
        if (
            isinstance(node.slice, ast.Index)
            or isinstance(node.slice, ast.Constant)
            or isinstance(node.slice, ast.Name)
        ):
            return IndexTransformer().to_custom_model(node)

        lower = super().to_custom_model(node.slice.lower)
        upper = super().to_custom_model(node.slice.upper)
        dependency_name = node.value.id

        return Subscript(node.lineno, dependency_name, lower, upper)

    def to_pymc(self, node: Subscript, conditions: dict, in_function):
        (operand, size) = self.program_variables[node.operand]
        lower = (
            super().to_pymc(node.lower, conditions, in_function) if node.lower else 0
        )
        upper = (
            super().to_pymc(node.upper, conditions, in_function) if node.upper else size
        )

        return operand[lower:upper]
