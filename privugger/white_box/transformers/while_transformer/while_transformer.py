from ...ast_transformer import AstTransformer
from ..if_transformer.if_model import If
from .while_model import While
import ast


class WhileTransformer(AstTransformer):
    def to_custom_model(self, node: ast.AST):
        body = super().collect_and_sort_by_line_number(node.body)
        test = super().to_custom_model(node.test)

        return While(node.lineno, test, body)

    def to_pymc(self, node: While, conditions, in_function):
        test = super().to_pymc(node.test, conditions, in_function)
        updated_conditions = conditions.copy()
        updated_conditions[node.name_with_line_number] = test
        while test.eval():
            for child_node in node.body:
                # Can't move this to the 'BreakTransformer' as 
                # it's only possible to break inside a loop.
                if (
                    isinstance(child_node, If)
                    and child_node.has_break_in_body
                    and child_node.name_with_line_number in conditions
                    and conditions[child_node.name_with_line_number].eval()
                ):
                    break

                else:
                    super().to_pymc(child_node, updated_conditions, in_function)

            test = super().to_pymc(node.test, updated_conditions, in_function)
