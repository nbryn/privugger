from ...ast_transformer import AstTransformer
import pytensor.tensor as pt
from .numpy_model import *
import pymc as pm
import ast


class NumpyTransformer(AstTransformer):
    numpy_to_custom_distribution_map = {
        "exponential": (NumpyExponential, ["scale"]),
        "laplace": (NumpyLaplace, ["loc", "scale"]),
        "uniform": (NumpyUniform, ["low", "high"]),
        "normal": (NumpyNormal, ["loc", "scale"]),
        "binomial": (NumpyBinomial, ["n", "p"]),
        "poisson": (NumpyPoisson, ["lam"]),
    }

    custom_distribution_to_pymc_map = {
        NumpyExponential: (pm.Exponential, ["scale"]),
        NumpyLaplace: (pm.Laplace, ["loc", "scale"]),
        NumpyUniform: (pm.Uniform, ["low", "high"]),
        NumpyNormal: (pm.Normal, ["loc", "scale"]),
        NumpyBinomial: (pm.Binomial, ["n", "p"]),
        NumpyPoisson: (pm.Poisson, ["lam"]),
    }

    def is_numpy(self, func):
        if isinstance(func, ast.Name):
            return False

        if isinstance(func.value, ast.Name):
            return func.value.id == "np"

        if isinstance(func.value.value, ast.Name):
            return func.value.value.id == "np"

        return False

    def to_custom_model(self, node: ast.Call):
        if self.__is_distribution(node.func.attr):
            return self.__handle_numpy_distribution(node)

        attribute = node.attr if hasattr(node, "attr") else node.func.attr
        np_operation = self.__to_custom_operation(attribute)
        mapped_arguments = list(map(super().to_custom_model, node.args))

        return NumpyFunction(node.lineno, np_operation, mapped_arguments)

    def __is_distribution(self, value):
        return any(
            value == distribution
            for distribution in self.numpy_to_custom_distribution_map.keys()
        )

    def __handle_numpy_distribution(self, node: ast.Call):
        if len(node.keywords) == 0:
            # Use np.random.exponential(scale=1) instead of np.random.exponential(1)
            raise TypeError(
                "Keyword arguments must be used when working with distributions"
            )

        if self.__find_argument(node, "size") != None:
            raise TypeError("The size keyword isn't supported yet")

        if node.func.attr not in self.numpy_to_custom_distribution_map:
            raise TypeError(f"Numpy distribution {node.func.attr} isn't supported yet")

        dist_class, params = self.numpy_to_custom_distribution_map[node.func.attr]
        args = [self.__find_argument(node, param) for param in params]

        return dist_class(node.lineno, *args)

    def __find_argument(self, node, keyword):
        argument = next((x.value for x in node.keywords if x.arg == keyword), None)
        return super().to_custom_model(argument)

    def __to_custom_operation(self, operation):
        if operation == "array":
            return NumpyOperation.ARRAY

        if operation == "exp":
            return NumpyOperation.EXP

        if operation == "dot":
            return NumpyOperation.DOT

        raise TypeError("Unknown numpy function")

    def to_pymc(self, node: Numpy, conditions: dict, in_function):
        if isinstance(node, NumpyFunction):
            mapped_arguments = list(map(super().to_pymc, node.arguments))
            if node.operation == NumpyOperation.ARRAY:
                return pt.as_tensor_variable(mapped_arguments[0])

            if node.operation == NumpyOperation.EXP:
                return pm.math.exp(mapped_arguments[0])

            if node.operation == NumpyOperation.DOT:
                return pm.math.dot(mapped_arguments[0][0], mapped_arguments[1][0])

            print(type(node))
            raise TypeError("Unknown Numpy function")

        return self.__to_pymc_distribution(node, conditions, in_function)

    def __to_pymc_distribution(self, node: Numpy, condition, in_function):
        for dist_type, (
            pymc_dist,
            params,
        ) in self.custom_distribution_to_pymc_map.items():
            if isinstance(node, dist_type):
                pymc_params = [
                    super(self.__class__, self).to_pymc(
                        getattr(node, param), condition, in_function
                    )
                    for param in params
                ]

                # We don't have access to the variable name here, so we have to return
                # a function that can be called in AssignTransformer (line 36)
                return lambda var_name: pymc_dist(var_name, *pymc_params)

        print(type(node))
        raise TypeError("Unknown Numpy distribution")
