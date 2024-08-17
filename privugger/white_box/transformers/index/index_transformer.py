from ...white_box_ast_transformer import WhiteBoxAstTransformer
from .index_model import Index
import ast

class IndexTransformer(WhiteBoxAstTransformer):
    def to_custom_model(self, node: ast.Index):
        return super().to_custom_model(node.value)
    
    def to_pymc(self, node: Index, _, __):
        (operand, _) = self.program_variables[node.operand]
        if isinstance(node.index, str):
            (index, _) = self.program_variables[node.index]
            return operand[index]

        return operand[node.index]