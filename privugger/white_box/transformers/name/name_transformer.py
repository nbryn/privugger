from .name_model import Reference
from .. import AstTransformer
import ast

class NameTransformer(AstTransformer):    
    def to_custom_model(self, node: ast.Name):
        return Reference(node.lineno, node.id)