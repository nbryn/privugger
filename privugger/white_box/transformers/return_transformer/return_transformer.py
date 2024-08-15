from .return_model import Return
from .. import AstTransformer
import pymc3 as pm
import ast


class ReturnTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Return):
        value = self._map_to_custom_type(node.value)
        return Return(node.lineno, value)

    def to_pymc(self, node: Return, pymc_model_builder, condition, in_function):
        value = pymc_model_builder.to_pymc(node.value, condition, in_function)
        if isinstance(value, tuple):
            value = value[0]

        if in_function:
            return value

        pm.Deterministic(node.name_with_line_number, value)
        return