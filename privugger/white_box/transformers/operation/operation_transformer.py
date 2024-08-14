from .operation_model import *
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