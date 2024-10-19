from ...ast_transformer import AstTransformer
from ..subscript.subscript_model import Subscript
from ..constant.constant_model import Constant
from ..list.list_model import List as ListModel
from ..name.name_model import Name
from collections.abc import Sized
from .assign_model import *
from functools import reduce
import pytensor.tensor as pt
import pymc as pm
import types
import ast


class AssignTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Assign):
        # Assumes only one target
        # IE: var1, var2 = 1, 2 not currently supported
        temp_node = node.targets[0]
        while not isinstance(temp_node, ast.Name):
            temp_node = temp_node.value

        if isinstance(node.targets[0], ast.Subscript):
            print("HERE")
            print(ast.dump(node))
            index = super().to_custom_model(node.targets[0].slice)
            value = super().to_custom_model(node.value)

            return AssignIndex(temp_node.id, node.lineno, value, index)

        value = super().to_custom_model(node.value)
        return Assign(temp_node.id, node.lineno, value)

    def to_pymc(self, node: Assign, conditions: dict, in_function):
        variable = super().to_pymc(node.value, conditions, in_function)
        if isinstance(variable, bool):
            self.program_variables[node.name] = (variable, -1)
            return

        # 'variable' is a function that returns a random variable
        if isinstance(variable, types.FunctionType):
            tensor_var = variable(node.name_with_line_number)
            self.program_variables[node.name] = (tensor_var, None)
            return tensor_var

        size = len(variable) if isinstance(variable, Sized) else None
        if isinstance(variable, tuple):
            variable = variable[0]

        # This handles assignment to parameter/argument.
        # We shouldn't (and can't) change the input as it's a distribution.
        if (
            isinstance(node.value, Name)
            and node.value.reference_to in self.program_arguments
        ):
            (operand, size) = self.program_variables[node.value.reference_to]
            variable = operand[: self.num_elements]

        pymc_variable_name = node.name_with_line_number
        tensor_var = pt.as_tensor_variable(variable)
        if isinstance(node, AssignIndex):
            index = super().to_pymc(node.index)
            if isinstance(index, tuple):
                index = index[0]

            (operand, _) = self.program_variables[node.name]
            tensor_var = pt.set_subtensor(operand[index], tensor_var)

            # If we mutate e.g. a list we need to replace the original list
            # but keep the variable name and line number.
            for var_name in self.pymc_model.named_vars:
                if var_name.startswith(node.name):
                    pymc_variable_name = var_name
                    break

            del self.pymc_model.named_vars[pymc_variable_name]

        # If 'conditions' isn't empty we're inside (nested) if/while statement(s).
        if len(conditions) > 0:
            # 'combined_conditions' handles nested if's.
            combined_conditions = reduce(pm.math.and_, list(conditions.values()))

            # Variable declared outside if
            # TODO: This doesn't work inside a loop as all occurrences
            # will set to value of the last time it was assigned.
            # Probably need to use variable with line number as name inside loop
            # Problem: First iteration program_variables[node.name_with_line_number] will not be present
            if node.name in self.program_variables:
                (current_var, size) = self.program_variables[node.name]
                tensor_var = pm.math.switch(
                    combined_conditions, tensor_var, current_var
                )

            # Variable declared inside .
            else:
                print("should not get here")
                (default_value, size) = self.__get_default_pymc_value(node.value)
                value = super().to_pymc(node.value)
                tensor_var = pm.math.switch(combined_conditions, value, default_value)

        self.program_variables[node.name] = (tensor_var, size)
        if not in_function:
            # Assignment inside loop to existing variable.
            if pymc_variable_name in self.pymc_model.named_vars:
                del self.pymc_model.named_vars[pymc_variable_name]

            pm.Deterministic(pymc_variable_name, tensor_var)

        return tensor_var

    # This method isn't needed if we constrain the input program as follows:
    # - Variables must be initialized outside if/while.
    def __get_default_pymc_value(self, node):
        if isinstance(node, Constant):
            if isinstance(node.value, int):
                return (pt.as_tensor_variable(0), None)

            if isinstance(node.value, float):
                return (pt.as_tensor_variable(0.0), None)

        if isinstance(node, ListModel):
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
