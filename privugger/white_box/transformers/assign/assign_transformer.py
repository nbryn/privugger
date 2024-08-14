from .. import ast_transformer
from .assign_model import *
import ast

class AssignTransformer(ast_transformer.AstTransformer):
    def __init__(self, ast_parser):
        super().__init__(ast_parser)
    
    def to_custom_model(self, node: ast.Assign):
        # Assumes only one target
        # IE: var1, var2 = 1, 2 not currently supported
        temp_node = node.targets[0]
        while not isinstance(temp_node, ast.Name):
            temp_node = temp_node.value

        if isinstance(node.targets[0], ast.Subscript):
            index = self.ast_parser.map_to_custom_type(node.targets[0].slice)
            value = self.ast_parser.map_to_custom_type(node.value)

            return AssignIndex(temp_node.id, node.lineno, value, index)

        value = self.ast_parser.map_to_custom_type(node.value)
        return Assign(temp_node.id, node.lineno, value)