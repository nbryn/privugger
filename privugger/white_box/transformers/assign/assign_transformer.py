from ..return_transformer.return_model import Return
from ..name.name_model import Name
from collections.abc import Sized
from ... import ast_transformer
from .assign_model import *
import pymc3.math as pm_math
import theano.tensor as tt
import pymc3 as pm

import ast

class AssignTransformer(ast_transformer.AstTransformer): 
    def to_custom_model(self, node: ast.Assign):
        # Assumes only one target
        # IE: var1, var2 = 1, 2 not currently supported
        temp_node = node.targets[0]
        while not isinstance(temp_node, ast.Name):
            temp_node = temp_node.value

        if isinstance(node.targets[0], ast.Subscript):
            index = self._map_to_custom_type(node.targets[0].slice)
            value = self._map_to_custom_type(node.value)

            return AssignIndex(temp_node.id, node.lineno, value, index)

        value = self._map_to_custom_type(node.value)
        return Assign(temp_node.id, node.lineno, value)
    
    def to_pymc(self, node: Assign, pymc_model_builder, condition, in_function):
        variable = pymc_model_builder.to_pymc(node.value, condition)
        size = len(variable) if isinstance(variable, Sized) else None
        if isinstance(variable, tuple):
            variable = variable[0]

        # This handles assignment to parameter/argument
        # We shouldn't (and can't) change the input as it's a distribution
        if (
            isinstance(node.value, Name)
            and node.value.reference_to in pymc_model_builder.program_arguments
        ):
            (operand, size) = pymc_model_builder.program_variables[node.value.reference_to]
            variable = operand[: pymc_model_builder.num_elements]

        pymc_variable_name = node.name_with_line_number
        tensor_var = tt.as_tensor_variable(variable)
        if isinstance(node, AssignIndex):
            (index, _) = pymc_model_builder.to_pymc(node.index)
            (operand, _) = pymc_model_builder.program_variables[node.name]
            tensor_var = tt.set_subtensor(operand[index], tensor_var)

            # If we mutate e.g. a list we need to replace the original list
            # but keep the variable name and line number
            for var_name in pymc_model_builder.pymc_model.named_vars:
                if var_name.startswith(node.name):
                    pymc_variable_name = var_name
                    break

            del pymc_model_builder.pymc_model.named_vars[pymc_variable_name]

        # Condition means we're inside if statement
        if condition:
            # Variable declared outside if
            if node.name in pymc_model_builder.program_variables:
                (current_var, size) = pymc_model_builder.program_variables[node.name]
                tensor_var = pm_math.switch(condition, tensor_var, current_var)

            # Variable declared inside if
            else:
                (default_value, size) = pymc_model_builder.get_default_value(node.value)
                value = pymc_model_builder.to_pymc(node.value)
                tensor_var = pm_math.switch(condition, value, default_value)

        pymc_model_builder.program_variables[node.name] = (tensor_var, size)

        if not in_function:
            pm.Deterministic(pymc_variable_name, tensor_var)

        return tensor_var