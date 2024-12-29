from ...ast_transformer import AstTransformer
from ..attribute.attribute_model import Attribute, AttributeOperation
from ..attribute.attribute_transformer import AttributeTransformer
from ..numpy.numpy_transformer import NumpyTransformer
from ..return_transformer.return_model import Return
from ..name.name_model import Name
from ..call.call_model import Call
import ast


class CallTransformer(AstTransformer):
    numpy_transformer = NumpyTransformer()

    def to_custom_model(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in [
            operation.name.lower() for operation in AttributeOperation
        ]:
            attribute = ast.Attribute(
                value=node.args[0], attr=node.func.id, lineno=node.lineno
            )
            return AttributeTransformer().to_custom_model(attribute)

        if self.numpy_transformer.is_numpy(node.func):
            return self.numpy_transformer.to_custom_model(node)

        operand = super().to_custom_model(node.func)
        mapped_arguments = list(map(super().to_custom_model, node.args))

        return Call(node.lineno, operand, mapped_arguments)

    def to_pymc(self, node: Call, conditions, in_function):
        if isinstance(node.operand, Attribute):
            return super().to_pymc(node.operand, conditions, in_function)

        mapped_arguments = list(map(super().to_pymc, node.arguments))
        if isinstance(node.operand, Name):
            return self.__handle_function_call(node, mapped_arguments, conditions)

        print(type(node.operand))
        raise TypeError("Unsupported call operand")

    def __handle_function_call(self, node, mapped_arguments, conditions):
        if node.operand.reference_to in self.program_functions:
            (function_body, function_arguments) = self.program_functions[
                node.operand.reference_to
            ]

            for index, argument_name in enumerate(function_arguments):
                self.program_variables[argument_name] = mapped_arguments[index]

            for child_node in function_body:
                if isinstance(child_node, Return):
                    return super().to_pymc(child_node, conditions, True)

                super().to_pymc(child_node, conditions, True)

        print(node.operand.reference_to)
        raise TypeError("Reference to unknown function")
