from .unaryop_model import UnaryOp
from .. import AstTransformer
import ast


class UnaryOpTransformer(AstTransformer):
    def to_custom_model(self, node: ast.UnaryOp):
        operation = self._map_operation(node.op)
        operand = self._map_to_custom_type(node.operand)

        return UnaryOp(node.lineno, operand, operation)

    def to_pymc(self, node: UnaryOp, pymc_model_builder, condition, in_function):
            operand = pymc_model_builder.to_pymc(node.operand, condition, in_function)
            if isinstance(operand, tuple):
                operand = operand[0]

            return pymc_model_builder.to_pymc_operation(node.operation, operand)
        