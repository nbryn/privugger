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
        if isinstance(node.func, ast.Name) and node.func.id in [operation.name.lower() for operation in AttributeOperation]:
            attribute = ast.Attribute(value=node.args[0], attr=node.func.id, lineno=node.lineno)
            return AttributeTransformer().to_custom_model(attribute)
            
        if self.numpy_transformer.is_numpy(node.func):
            return self.numpy_transformer.to_custom_model(node)

        # TODO: Can operand both be a function and an object?
        operand = super().to_custom_model(node.func)
        mapped_arguments = list(map(super().to_custom_model, node.args))

        return Call(node.lineno, operand, mapped_arguments)

    def to_pymc(self, node: Call, condition, in_function):
        if isinstance(node.operand, Attribute):
            return super().to_pymc(node.operand, condition, in_function)

        mapped_arguments = list(map(super().to_pymc, node.arguments))

        if isinstance(node.operand, Name):
            # TODO: Function call - Extract to handle_function_call
            if node.operand.reference_to in self.program_functions:
                (function_body, function_arguments) = self.program_functions[
                    node.operand.reference_to
                ]
                # TODO: Execute function body. Variables should NOT be added to program_variables
                # And return should actually return the value instead of creating program variable

                for index, argument_name in enumerate(function_arguments):
                    self.program_variables[argument_name] = mapped_arguments[index]

                # TODO: This doesn't handle function with multiple returns (returns in if)
                # as it return the first time a return node is encountered on top level.
                # Returns inside ifs in functions should be handled in __handle_if
                for child_node in function_body:
                    if isinstance(child_node, Return):
                        return super().to_pymc(child_node, None, True)

                    super().to_pymc(child_node, None, True)

            # TODO: Will it always be a reference to a function?
            print(node.operand.reference_to)
            raise TypeError("Reference to unknown function")

        print(type(node.operand))
        raise TypeError("Unsupported call operand")
