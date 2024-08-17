from .operation_model import *
import pymc3.math as pm_math
import theano.tensor as tt
import ast

class OperationTransformer():
    def to_custom_model(self, operation):
        if operation == "sum":
            return Operation.SUM

        if operation == "size" or operation == "len":
            return Operation.SIZE

        if isinstance(operation, ast.Add) or isinstance(operation, ast.UAdd):
            return Operation.ADD

        if isinstance(operation, ast.Sub) or isinstance(operation, ast.USub):
            return Operation.SUB

        if isinstance(operation, ast.Div):
            return Operation.DIVIDE

        if isinstance(operation, ast.Mult):
            return Operation.MULTIPLY

        if isinstance(operation, ast.Lt):
            return Operation.LT

        if isinstance(operation, ast.LtE):
            return Operation.LTE

        if isinstance(operation, ast.Gt):
            return Operation.GT

        if isinstance(operation, ast.GtE):
            return Operation.GTE

        print(operation)
        raise TypeError("Unknown operation")
    
    def to_pymc(self, operation: Operation, operand, right=None):
        if operation == Operation.EQUAL:
            return pm_math.eq(operand, right)

        if operation == Operation.LT:
            return pm_math.lt(operand, right)

        if operation == Operation.LTE:
            return pm_math.le(operand, right)

        if operation == Operation.GT:
            return pm_math.gt(operand, right)

        if operation == Operation.GTE:
            return pm_math.ge(operand, right)

        if operation == Operation.DIVIDE:
            return operand / right

        if operation == Operation.SUB:
            if right:
                return operand - right

            return -operand

        if operation == Operation.ADD:
            if right:
                return operand + right

            return +operand

        if operation == Operation.SUM:
            return (
                pm_math.sum(operand)
                if isinstance(operand, tt.TensorVariable)
                else sum(operand)
            )

        if operation == Operation.SIZE:
            return (
                operand.shape[0]
                if isinstance(operand, tt.TensorVariable)
                else len(operand)
            )

        print(operation)
        raise TypeError("Unsupported operation")
