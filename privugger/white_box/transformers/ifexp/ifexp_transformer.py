from ...ast_transformer import AstTransformer
from .ifexp_model import IfExp
import pymc as pm
import ast


class IfExpTransformer(AstTransformer):
    def to_custom_model(self, node: ast.IfExp):
        condition = super().to_custom_model(node.test)
        orelse = super().to_custom_model(node.orelse)
        body = super().to_custom_model(node.body)

        return IfExp(node.lineno, condition, body, orelse)

    def to_pymc(self, node: IfExp, existing_condition, in_function):
        condition = super().to_pymc(node.condition, existing_condition, in_function)
        if_false = super().to_pymc(node.orelse, existing_condition, in_function)
        if_true = super().to_pymc(node.body, existing_condition, in_function)

        return pm.math.switch(condition, if_true, if_false)
