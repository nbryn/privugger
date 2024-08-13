from .. import base_transformer
from .numpy_model import *
import ast


class NumpyTransformer(base_transformer.Transformer):
    def __init__(self, ast_transformer):
        base_transformer.Transformer.__init__(self, ast_transformer)

    def is_numpy(self, func):
        if isinstance(func.value, ast.Name):
            return func.value.id == "np"

        if isinstance(func.value.value, ast.Name):
            return func.value.value.id == "np"

        return False

    def transform(self, node: ast.Call):
        if self.__is_distribution(node.func.attr):
            return self.__handle_numpy_distribution(node)

        np_operation = self.__map_numpy_operation(node.attr)
        mapped_arguments = list(map(self.ast_transformer.map_to_custom_type, node.args))
        return Function(node.lineno, np_operation, mapped_arguments)

    def __is_distribution(self, value):
        return any(
            value == distribution.value.lower() for distribution in DistributionType
        )

    def __handle_numpy_distribution(self, node: ast.Call):
        loc = self.ast_transformer.map_to_custom_type(node.keywords[0].value)
        scale = self.ast_transformer.map_to_custom_type(node.keywords[1].value)

        if node.func.attr == "normal":
            return Distribution(node.lineno, DistributionType.NORMAL, scale, loc)

        if node.func.attr == "laplace":
            return Distribution(node.lineno, DistributionType.LAPLACE, scale, loc)

        raise TypeError("Unknown numpy distribution")

    def __map_numpy_operation(self, operation):
        if operation == "array":
            return Operation.ARRAY

        if operation == "exp":
            return Operation.EXP

        if operation == "dot":
            return Operation.DOT

        raise TypeError("Unknown numpy function")
