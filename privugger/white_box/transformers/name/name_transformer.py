from ...white_box_ast_transformer import WhiteBoxAstTransformer
from .name_model import Name
import ast


class NameTransformer(WhiteBoxAstTransformer):
    def to_custom_model(self, node: ast.Name):
        return Name(node.lineno, node.id)

    def to_pymc(self, node: Name, _, __):
        return self.program_variables[node.reference_to]
