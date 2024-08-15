from .name_model import Name
from .. import AstTransformer
import ast

class NameTransformer(AstTransformer):    
    def to_custom_model(self, node: ast.Name):
        return Name(node.lineno, node.id)
    
    def to_pymc(self, node: Name, pymc_model_builder, condition, in_function):
        return pymc_model_builder.program_variables[node.reference_to]