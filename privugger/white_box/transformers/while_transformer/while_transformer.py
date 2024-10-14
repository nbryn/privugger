from ...ast_transformer import AstTransformer
from .while_model import While
import ast


class WhileTransformer(AstTransformer):
    def to_custom_model(self, node: ast.AST):
        body = super().collect_and_sort_by_line_number(node.body)
        test = super().to_custom_model(node.test)

        return While(node.lineno, test, body)

    def to_pymc(self, node: While, condition, in_function):
        test = super().to_pymc(node.test, condition, in_function)
        while test.eval():
            for child_node in node.body:
                super().to_pymc(child_node, test, in_function)

            test = super().to_pymc(node.test, test, in_function)
