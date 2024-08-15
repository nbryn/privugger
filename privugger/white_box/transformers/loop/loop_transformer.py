from ..constant.constant_model import Constant
from .. import AstTransformer
from .loop_model import Loop
import ast


class ForTransformer(AstTransformer):
    def to_custom_model(self, node: ast.For):
        # Loops are not well supported in PyMC, meaning they shouldn't be translated
        # into PyMC variables but instead function as a 'normal' python loop.
        # Only standard 'for i in range()' loops supported for now
        body = self._collect_and_sort_by_line_number(node.body)

        # We only have stop like 'range(stop)'
        if len(node.iter.args) == 1:
            start = Constant(node.lineno, 0)
            stop = self._map_to_custom_type(node.iter.args[0])
            return Loop(node.lineno, start, stop, body)

        # We have both start and stop like 'range(start, stop)'
        start = self._map_to_custom_type(node.iter.args[0])
        stop = self._map_to_custom_type(node.iter.args[1])

        return Loop(node.lineno, start, stop, body)
