from ...white_box_ast_transformer import WhiteBoxAstTransformer
from .if_model import If
import ast


class IfTransformer(WhiteBoxAstTransformer):
    def to_custom_model(self, node: ast.If):
        # TODO: Move to 'handle_if' method
        # Note 'node.orelse' is not handled here but as a separate node
        body_custom_nodes = super().collect_and_sort_by_line_number(node.body)
        condition = super().to_custom_model(node.test)

        return If(node.lineno, condition, body_custom_nodes)

    def to_pymc(self, node: If, condition, in_function):
        condition = super().to_pymc(node.condition, condition, in_function)
        for child_node in node.body:
            super().to_pymc(child_node, condition, condition, in_function)
