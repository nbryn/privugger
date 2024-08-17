from ...white_box_ast_transformer import WhiteBoxAstTransformer
from .unaryop_model import UnaryOp
import ast


class UnaryOpTransformer(WhiteBoxAstTransformer):
    def to_custom_model(self, node: ast.UnaryOp):
        operation = super()._map_operation(node.op)
        operand = super().to_custom_model(node.operand)

        return UnaryOp(node.lineno, operand, operation)

    def to_pymc(self, node: UnaryOp, condition, in_function):
        operand = super().to_pymc(node.operand, condition, in_function)
        if isinstance(operand, tuple):
            operand = operand[0]

        return super()._to_pymc_operation(node.operation, operand)
