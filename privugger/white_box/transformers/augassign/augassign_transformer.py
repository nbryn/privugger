from ...ast_transformer import AstTransformer
from .augassign_model import AugAssign, AugAssignOperation
import pytensor.tensor as pt
import ast


class AugAssignTransformer(AstTransformer):
    def to_custom_model(self, node: ast.AugAssign):
        operation = self.__to_custom_operation(node.op)
        operand = super().to_custom_model(node.target)
        value = super().to_custom_model(node.value)

        return AugAssign(node.lineno, operand, operation, value)

    def to_pymc(self, node: AugAssign, condition, in_function):
        operand = self.program_variables[node.operand.reference_to]
        value = super().to_pymc(node.value, condition, in_function)
        
        if isinstance(operand, tuple):
            operand = operand[0]
            
        if node.operation == AugAssignOperation.ADD:
            updated = operand + pt.as_tensor_variable(value)
            self.program_variables[node.operand.reference_to] = updated
            return

        if node.operation == AugAssignOperation.SUB:
            updated = operand - pt.as_tensor_variable(value)
            self.program_variables[node.operand.reference_to] = updated 
            return

        print(node.operation)
        raise TypeError("Unsupported AugAssign operation")

    def __to_custom_operation(self, operation: ast):
        if isinstance(operation, ast.Add):
            return AugAssignOperation.ADD

        if isinstance(operation, ast.Sub):
            return AugAssignOperation.SUB

        print(operation)
        raise TypeError("Unsupported AugAssign operation")
