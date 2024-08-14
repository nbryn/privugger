from .. import AstTransformer
from .numpy_model import *
import ast


class NumpyTransformer(AstTransformer):
    def is_numpy(self, func):
        if isinstance(func.value, ast.Name):
            return func.value.id == "np"

        if isinstance(func.value.value, ast.Name):
            return func.value.value.id == "np"

        return False

    def to_custom_model(self, node: ast.Call):
        if self.__is_distribution(node.func.attr):
            return self.__handle_numpy_distribution(node)

        np_operation = self.__map_numpy_operation(node.attr)
        mapped_arguments = list(map(self.map_to_custom_type, node.args))
        return NumpyFunction(node.lineno, np_operation, mapped_arguments)

    def __is_distribution(self, value):
        return any(
            value == distribution.value.lower() for distribution in NumpyDistributionType
        )

    def __handle_numpy_distribution(self, node: ast.Call):
        loc = self.map_to_custom_type(node.keywords[0].value)
        scale = self.map_to_custom_type(node.keywords[1].value)

        if node.func.attr == "normal":
            return NumpyDistribution(node.lineno, NumpyDistributionType.NORMAL, scale, loc)

        if node.func.attr == "laplace":
            return NumpyDistribution(node.lineno, NumpyDistributionType.LAPLACE, scale, loc)

        raise TypeError("Unknown numpy distribution")

    def __map_numpy_operation(self, operation):
        if operation == "array":
            return NumpyOperation.ARRAY

        if operation == "exp":
            return NumpyOperation.EXP

        if operation == "dot":
            return NumpyOperation.DOT

        raise TypeError("Unknown numpy function")
