from ...ast_transformer import AstTransformer
from .index_model import Index
import ast

class IndexTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Index):
        return self._map_to_custom_type(node.value)
    
    def to_pymc(self, node: Index, pymc_model_builder, condition, in_function):
        (operand, _) = pymc_model_builder.program_variables[node.operand]
        if isinstance(node.index, str):
            (index, _) = pymc_model_builder.program_variables[node.index]
            return operand[index]

        return operand[node.index]