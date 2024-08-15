from .. import AstTransformer, NumpyTransformer
from ..call.call_model import Call
import ast


class CallTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Call):
        numpy_transformer = NumpyTransformer()
        if numpy_transformer.is_numpy(node.func):
            return numpy_transformer.to_custom_model(node)

        # TODO: Can operand both be a function and an object?
        operand = self._map_to_custom_type(node.func)
        mapped_arguments = list(map(self._map_to_custom_type, node.args))

        return Call(node.lineno, operand, mapped_arguments)
