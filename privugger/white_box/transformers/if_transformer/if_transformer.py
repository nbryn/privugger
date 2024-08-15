from .. import AstTransformer
import ast

class IfTransformer(AstTransformer):
    def __init__(self, ast_parser):
        super().__init__(ast_parser)
    
    def to_custom_model(self, node: ast.If):
          # TODO: Move to 'handle_if' method
            # Note 'node.orelse' is not handled here but as a separate node
            body_custom_nodes = self._collect_and_sort_by_line_number(node.body)
            condition = self._map_to_custom_type(node.test)

            return If(node.lineno, condition, body_custom_nodes)