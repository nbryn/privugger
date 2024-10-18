from ...ast_transformer import AstTransformer
from .index_model import Index
from typing import Union
import ast


class IndexTransformer(AstTransformer):
    def to_custom_model(self, node: Union[ast.Index, ast.Subscript]):
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant):
                index = node.slice.value
                return Index(node.lineno, node.value.id, index)

            if isinstance(node.slice, ast.Name):
                return Index(node.lineno, node.value.id, node.slice.id)

            index = (
                node.slice.value.value
                if hasattr(node.value, "value")
                else node.slice.value.id
            )

            return Index(node.lineno, node.value.id, index)

        return super().to_custom_model(node.value)

    def to_pymc(self, node: Index, _, __):
        (operand, _) = self.program_variables[node.operand]
        if isinstance(node.index, str):
            (index, _) = self.program_variables[node.index]
            return operand[index]

        return operand[node.index]
