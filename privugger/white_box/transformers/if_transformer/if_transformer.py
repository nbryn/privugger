from ...ast_transformer import AstTransformer
from .if_model import If
import ast


class IfTransformer(AstTransformer):
    def to_custom_model(self, node: ast.If):
        orelse = super().collect_and_sort_by_line_number(node.orelse)
        body = super().collect_and_sort_by_line_number(node.body)
        condition = super().to_custom_model(node.test)

        return If(node.lineno, condition, body, orelse)

    # TODO: This probably doesn't handle nested if statements.
    # TODO: This doesn't handle 'elif'.
    # AssignTransformer handles conditionally assigning values
    # depending on whether the condition is true or not.
    def to_pymc(self, node: If, existing_condition, in_function):
        condition = super().to_pymc(node.condition, existing_condition, in_function)
        for child_node in node.body + node.orelse:
            super().to_pymc(child_node, condition, in_function)
