from .. import base_transformer
from .binop_model import *

class BinOpTransformer(base_transformer.Transformer):
    def __init__(self, ast_transformer):
        base_transformer.Transformer.__init__(self, ast_transformer)
    
    def transform(self, node):
        operation = self.ast_transformer.map_operation(node.op)
        right = self.ast_transformer.map_to_custom_type(node.right)
        left = self.ast_transformer.map_to_custom_type(node.left)

        return model.BinOp(node.lineno, left, right, operation) 
    
    