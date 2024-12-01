from ...ast_transformer import AstTransformer
from .index_model import Index
import ast


class IndexTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Index | ast.Subscript):
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Subscript):
                first_index = self.__get_custom_model_index(node.value)
                second_index = self.__get_custom_model_index(node)
                return Index(
                    node.lineno, node.value.value.id, first_index, second_index
                )

            index = self.__get_custom_model_index(node)
            return Index(node.lineno, node.value.id, index)

        return super().to_custom_model(node.value)

    def to_pymc(self, node: Index, _, __):
        (operand, _) = self.program_variables[node.operand]
        if node.second_index:
            (first_index, _) = self.__get_pymc_index(node.first_index)
            (second_index, _) = self.__get_pymc_index(node.second_index)
            return operand[first_index][second_index]

        index = self.__get_pymc_index(node.first_index)
        return operand[index]

    def __get_custom_model_index(self, node: ast):
        if isinstance(node.slice, ast.Constant):
            return node.slice.value

        if isinstance(node.slice, ast.Name):
            return node.slice.id

        return (
            node.slice.value.value
            if hasattr(node.value, "value")
            else node.slice.value.id
        )

    def __get_pymc_index(self, index):
        if isinstance(index, str):
            (index, _) = self.program_variables[index]
            return index

        return index
