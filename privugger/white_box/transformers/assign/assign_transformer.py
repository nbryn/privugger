from ...ast_transformer import AstTransformer
from ..subscript.subscript_model import Subscript
from ..constant.constant_model import Constant
from ..list.list_model import ListNode
from ..name.name_model import Name
from collections.abc import Sized
from .assign_model import *
import pytensor.tensor as pt
import pymc as pm
import ast


class AssignTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Assign):
        # Assumes only one target
        # IE: var1, var2 = 1, 2 not currently supported
        temp_node = node.targets[0]
        while not isinstance(temp_node, ast.Name):
            temp_node = temp_node.value

        if isinstance(node.targets[0], ast.Subscript):
            index = super().to_custom_model(node.targets[0].slice)
            value = super().to_custom_model(node.value)

            return AssignIndex(temp_node.id, node.lineno, value, index)

        value = super().to_custom_model(node.value)
        return Assign(temp_node.id, node.lineno, value)

    def to_pymc(self, node: Assign, condition, in_function):
        print(node.value)
        variable = super().to_pymc(node.value, condition, in_function)
        if isinstance(variable, bool):
            self.program_variables[node.name] = (variable, -1)
            return

        size = len(variable) if isinstance(variable, Sized) else None
        if isinstance(variable, tuple):
            variable = variable[0]

        # This handles assignment to parameter/argument
        # We shouldn't (and can't) change the input as it's a distribution
        if (
            isinstance(node.value, Name)
            and node.value.reference_to in self.program_arguments
        ):
            (operand, size) = self.program_variables[node.value.reference_to]
            variable = operand[: self.num_elements]

        pymc_variable_name = node.name_with_line_number
        tensor_var = pt.as_tensor_variable(variable)
        if isinstance(node, AssignIndex):
            (index, _) = super().to_pymc(node.index)
            (operand, _) = self.program_variables[node.name]
            tensor_var = pt.set_subtensor(operand[index], tensor_var)

            # If we mutate e.g. a list we need to replace the original list
            # but keep the variable name and line number
            for var_name in self.pymc_model.named_vars:
                if var_name.startswith(node.name):
                    pymc_variable_name = var_name
                    break

            del self.pymc_model.named_vars[pymc_variable_name]

        # Condition means we're inside if statement
        # TODO: Move to if_transformer?
        if condition:
            # Variable declared outside if
            if node.name in self.program_variables:
                (current_var, size) = self.program_variables[node.name]
                tensor_var = pm.math.switch(condition, tensor_var, current_var)

            # Variable declared inside if
            else:
                (default_value, size) = self.__get_default_pymc_value(node.value)
                value = super().to_pymc(node.value)
                tensor_var = pm.math.switch(condition, value, default_value)

        self.program_variables[node.name] = (tensor_var, size)

        if not in_function:
            pm.Deterministic(pymc_variable_name, tensor_var)

        return tensor_var

    # This method isn't needed if we constrain input program as follows:
    # - Variables must be initialized outside if
    def __get_default_pymc_value(self, node):
        if isinstance(node, Constant):
            if isinstance(node.value, int):
                return (pt.as_tensor_variable(0), None)

            if isinstance(node.value, float):
                return (pt.as_tensor_variable(0.0), None)

        if isinstance(node, ListNode):
            if len(node.values) == 0:
                return ([], 0)

            return (
                pt.as_tensor_variable([pt.as_tensor_variable(0)]) * len(node.values),
                len(node.values),
            )

        if isinstance(node, Subscript):
            (operand, _) = self.program_variables[node.operand]
            lower = super().to_pymc(node.lower) if node.lower else 0
            upper = super().to_pymc(node.upper) if node.upper else len(operand)

            # TODO: Check if continuous or discrete?
            return (
                pt.as_tensor_variable([pt.as_tensor_variable(0)] * (upper - lower)),
                upper - lower,
            )

        # Attribute/Call etc will always be the default value no matter the operation
        # Distinguish between float and int?
        return (pt.as_tensor_variable(0), None)
