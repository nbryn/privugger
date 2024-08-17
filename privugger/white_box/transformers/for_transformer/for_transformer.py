from ...white_box_ast_transformer import WhiteBoxAstTransformer
from ..constant.constant_model import Constant
from .for_model import For
import ast


class ForTransformer(WhiteBoxAstTransformer):
    def to_custom_model(self, node: ast.For):
        # Loops are not well supported in PyMC, meaning they shouldn't be translated
        # into PyMC variables but instead function as a 'normal' python loop.
        # Only standard 'for i in range()' loops supported for now
        
        body = super().collect_and_sort_by_line_number(node.body)

        # We only have stop like 'range(stop)'
        if len(node.iter.args) == 1:
            start = Constant(node.lineno, 0)
            stop = super().to_custom_model(node.iter.args[0])
            return For(node.lineno, start, stop, body)

        # We have both start and stop like 'range(start, stop)'
        start = super().to_custom_model(node.iter.args[0])
        stop = super().to_custom_model(node.iter.args[1])

        return For(node.lineno, start, stop, body)

    def to_pymc(self, node: For, condition, in_function):
        start = super().to_pymc(node.start)
        stop = super().to_pymc(node.stop)

        for i in range(start, stop):
            self.program_variables["i"] = (i, None)
            for child_node in node.body:
                super().to_pymc(child_node)

        # Below is kept for future reference.
        # TODO: Verify that this works as expected
        """  outputs, updates = scan(fn=loop_body, sequences=tt.arange(start, stop))
            def loop_body(i, *args):
                self.model_variables['i'] = (i, None)
                for child_node in node.body:
                    self.__map_to_pycm_var(child_node)
                return [] """

        # If necessary, handle the outputs and updates here.
        # For example, you could create a Deterministic variable to store the outputs.
        # pm.Deterministic('loop_outputs', outputs)

        # return outputs, updates