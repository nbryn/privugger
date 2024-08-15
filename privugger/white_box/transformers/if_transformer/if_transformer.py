from .. import AstTransformer
from .if_model import If
import ast

class IfTransformer(AstTransformer):
    def to_custom_model(self, node: ast.If):
        # TODO: Move to 'handle_if' method
        # Note 'node.orelse' is not handled here but as a separate node
        body_custom_nodes = self._collect_and_sort_by_line_number(node.body)
        condition = self._map_to_custom_type(node.test)
    
        return If(node.lineno, condition, body_custom_nodes)
    
    def to_pymc(self, node: If, pymc_model_builder):
        condition = pymc_model_builder.to_pymc(node.condition)
        for child_node in node.body:
            pymc_model_builder.to_pymc(child_node, condition)