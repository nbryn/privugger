from .. import AstTransformer
import pymc3.math as pm_math
from .numpy_model import *
import pymc3 as pm
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

        np_operation = self.__to_custom_operation(node.attr)
        mapped_arguments = list(map(self._map_to_custom_type, node.args))
        return NumpyFunction(node.lineno, np_operation, mapped_arguments)

    def __is_distribution(self, value):
        return any(
            value == distribution.value.lower()
            for distribution in NumpyDistributionType
        )

    def __handle_numpy_distribution(self, node: ast.Call):
        loc = self._map_to_custom_type(node.keywords[0].value)
        scale = self._map_to_custom_type(node.keywords[1].value)

        if node.func.attr == "normal":
            return NumpyDistribution(
                node.lineno, NumpyDistributionType.NORMAL, scale, loc
            )

        if node.func.attr == "laplace":
            return NumpyDistribution(
                node.lineno, NumpyDistributionType.LAPLACE, scale, loc
            )

        raise TypeError("Unknown numpy distribution")

    def __to_custom_operation(self, operation):
        if operation == "array":
            return NumpyOperation.ARRAY

        if operation == "exp":
            return NumpyOperation.EXP

        if operation == "dot":
            return NumpyOperation.DOT

        raise TypeError("Unknown numpy function")

    def to_pymc(self, node: Numpy, model_builder, condition, in_function):
        if isinstance(node, NumpyFunction):
            mapped_arguments = list(map(model_builder.to_pymc, node.arguments))
            if node.operation == NumpyOperation.ARRAY:
                return mapped_arguments[0]

            if node.operation == NumpyOperation.EXP:
                return pm_math.exp(mapped_arguments[0])

            if node.operation == NumpyOperation.DOT:
                return pm_math.dot(mapped_arguments[0][0], mapped_arguments[1][0])

            print(type(node))
            raise TypeError("Unknown numpy operation")

        if isinstance(node, NumpyDistribution):
            loc = model_builder.to_pymc(node.loc, condition, in_function)
            scale = model_builder.to_pymc(node.scale, condition, in_function)
            if node.distribution == NumpyDistributionType.NORMAL:
                return pm.Normal(node.name_with_line_number, loc, scale)

            if node.distribution == NumpyDistributionType.LAPLACE:
                return pm.Laplace(node.name_with_line_number, loc, scale)

            print(type(node))
            raise TypeError("Unknown numpy distribution")
