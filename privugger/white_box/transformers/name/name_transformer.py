from .name_model import Reference
from .. import AstTransformer
import ast

class NameTransformer(AstTransformer):
    def __init__(self, ast_parser):
        super().__init__(ast_parser)
    
    def to_custom_model(self, node: ast.Name):
        return Reference(node.lineno, node.id)