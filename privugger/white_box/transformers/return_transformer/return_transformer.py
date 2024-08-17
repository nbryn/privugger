from ...white_box_ast_transformer import WhiteBoxAstTransformer
from .return_model import Return
import pymc3 as pm
import ast


class ReturnTransformer(WhiteBoxAstTransformer):
    def to_custom_model(self, node: ast.Return):
        value = super().to_custom_model(node.value)
        return Return(node.lineno, value)

    def to_pymc(self, node: Return, condition, in_function):
        value = super().to_pymc(node.value, condition, in_function)
        if isinstance(value, tuple):
            value = value[0]

        if in_function:
            return value

        pm.Deterministic(node.name_with_line_number, value)
        return
