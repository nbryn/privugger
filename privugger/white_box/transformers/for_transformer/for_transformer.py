from ...ast_transformer import AstTransformer
from ..break_transformer.break_transformer import BreakTransformer
from ..constant.constant_model import Constant
from ..if_transformer.if_model import If
from .for_model import For
import ast


# Loops aren't well supported in PyMC, meaning they shouldn't be translated
# into something PyMC specific, but instead function as a normal python loop.
# Only standard 'for i in range()' for loops supported atm.
class ForTransformer(AstTransformer):
    break_transformer = BreakTransformer()

    def to_custom_model(self, node: ast.For):
        body = super().collect_and_sort_by_line_number(node.body)
        loop_var = node.target.id

        # We only have stop as in 'range(stop)'.
        if len(node.iter.args) == 1:
            start = Constant(node.lineno, 0)
            stop = super().to_custom_model(node.iter.args[0])
            return For(node.lineno, loop_var, start, stop, body)

        # We have both start and stop as in 'range(start, stop)'.
        start = super().to_custom_model(node.iter.args[0])
        stop = super().to_custom_model(node.iter.args[1])

        return For(node.lineno, loop_var, start, stop, body)

    def to_pymc(self, node: For, conditions: dict, in_function):
        start = super().to_pymc(node.start)
        stop = super().to_pymc(node.stop)
        if isinstance(start, tuple):
            start = start[0]

        if isinstance(stop, tuple):
            stop = stop[0]

        should_break = False
        for i in range(start, stop):
            self.program_variables[node.loop_var] = (i, None)
            for child_node in node.body:
                should_break = self.break_transformer.should_break(child_node)
                super().to_pymc(child_node, conditions, in_function)

            if should_break:
                break
