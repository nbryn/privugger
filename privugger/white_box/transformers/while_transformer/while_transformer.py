from ...ast_transformer import AstTransformer
from ..break_transformer.break_transformer import BreakTransformer
from ..if_transformer.if_model import If
from .while_model import While
import ast


class WhileTransformer(AstTransformer):
    break_transformer = BreakTransformer()

    def to_custom_model(self, node: ast.AST):
        body = super().collect_and_sort_by_line_number(node.body)
        test = super().to_custom_model(node.test)

        return While(node.lineno, test, body)

    def to_pymc(self, node: While, conditions, in_function):
        test = super().to_pymc(node.test, conditions, in_function)
        updated_conditions = conditions.copy()
        updated_conditions[node.name_with_line_number] = test

        should_break = False
        while test.eval():
            for child_node in node.body:
                should_break = self.break_transformer.should_break(child_node)
                super().to_pymc(child_node, conditions, in_function)

            if should_break:
                break

            test = super().to_pymc(node.test, updated_conditions, in_function)
