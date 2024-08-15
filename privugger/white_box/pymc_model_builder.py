from .transformers import *
from collections.abc import Sized
import pymc3.math as pm_math
import theano.tensor as tt
from theano.scan import scan
from typing import List
import pymc3 as pm


# Make methods and fields private
class PyMCModelBuilder:
    custom_nodes: List[CustomNode] = []
    program_arguments = []
    program_variables = {}
    program_functions = {}
    global_priors = []
    num_elements = 0
    pymc_model = None

    def __init__(
        self, custom_nodes, program_arguments, global_priors, num_elements, pymc_model
    ):
        self.program_arguments = program_arguments
        self.global_priors = global_priors
        self.num_elements = num_elements
        self.custom_nodes = custom_nodes
        self.pymc_model = pymc_model

    def build(self):
        # Map top level function args to PyMC. Args must be of type 'pv.Distribution'
        for index, arg_name in enumerate(self.program_arguments):
            # TODO: Can we have more than one argument?
            self.program_variables[arg_name] = (
                self.global_priors[index],
                self.num_elements,
            )

        for node in self.custom_nodes:
            self.to_pymc(node)

    def to_pymc(self, node: CustomNode, condition=None, in_function=False):
        # print("MAP TO")
        # print(type(node))
        if isinstance(node, Subscript):
            return self.__handle_subscript(node)

        if isinstance(node, Constant):
            return node.value

        if isinstance(node, Compare) or isinstance(node, Compare2):
            return self.__handle_compare(node)

        if isinstance(node, Index):
            return self.__handle_index(node)

        if isinstance(node, UnaryOp):
            operand = self.to_pymc(node.operand, condition, in_function)
            if isinstance(operand, tuple):
                operand = operand[0]

            return self.__to_pymc_operation(node.operation, operand)

        if isinstance(node, BinOp):
            return self.__handle_binop(node)

        if isinstance(node, Reference):
            return self.program_variables[node.reference_to]

        if isinstance(node, Return):
            value = self.to_pymc(node.value, condition, in_function)
            if isinstance(value, tuple):
                value = value[0]

            if in_function:
                return value

            pm.Deterministic(node.name_with_line_number, value)
            return

        if isinstance(node, Assign):
            return self.__handle_assign(node, condition)

        if isinstance(node, Attribute):
            return self.__handle_attribute(node)

        if isinstance(node, Call):
            return self.__handle_call(node)

        if isinstance(node, Loop):
            return self.__handle_loop(node)

        if isinstance(node, If):
            return self.__handle_if(node)

        if isinstance(node, ListNode):
            return list(map(self.to_pymc, node.values))

        if isinstance(node, FunctionDef):
            self.program_functions[node.name] = (node.body, node.arguments)
            return

        if hasattr(node, "to_pymc"):
            node.to_pymc(self)

        print(type(node))
        raise TypeError("Unsupported custom ast type")

    def __handle_assign(self, node: Assign, condition=None, in_function=False):
        variable = self.to_pymc(node.value, condition)
        size = len(variable) if isinstance(variable, Sized) else None
        if isinstance(variable, tuple):
            variable = variable[0]

        # This handles assignment to parameter/argument
        # We shouldn't (and can't) change the input as it's a distribution
        if (
            isinstance(node.value, Reference)
            and node.value.reference_to in self.program_arguments
        ):
            (operand, size) = self.program_variables[node.value.reference_to]
            variable = operand[: self.num_elements]

        pymc_variable_name = node.name_with_line_number
        tensor_var = tt.as_tensor_variable(variable)
        if isinstance(node, AssignIndex):
            (index, _) = self.to_pymc(node.index)
            (operand, _) = self.program_variables[node.name]
            tensor_var = tt.set_subtensor(operand[index], tensor_var)

            # If we mutate e.g. a list we need to replace the original list
            # but keep the variable name and line number
            for var_name in self.pymc_model.named_vars:
                if var_name.startswith(node.name):
                    pymc_variable_name = var_name
                    break

            del self.pymc_model.named_vars[pymc_variable_name]

        # Condition means we're inside if statement
        if condition:
            # Variable declared outside if
            if node.name in self.program_variables:
                (current_var, size) = self.program_variables[node.name]
                tensor_var = pm_math.switch(condition, tensor_var, current_var)

            # Variable declared inside if
            else:
                (default_value, size) = self.__get_default_value(node.value)
                value = self.to_pymc(node.value)
                tensor_var = pm_math.switch(condition, value, default_value)

        self.program_variables[node.name] = (tensor_var, size)

        if not in_function:
            pm.Deterministic(pymc_variable_name, tensor_var)

        return tensor_var

    def __handle_subscript(self, node: Subscript):
        (operand, size) = self.program_variables[node.operand]
        lower = self.to_pymc(node.lower) if node.lower else 0
        upper = self.to_pymc(node.upper) if node.upper else size

        return operand[lower:upper]

    def __handle_compare(self, node):
        left = self.to_pymc(node.left)
        right = self.to_pymc(node.right)

        if isinstance(left, tuple):
            left = left[0]

        if isinstance(node, Compare):
            return self.__to_pymc_operation(node.operation, left, right)

        middle = self.to_pymc(node.middle)
        left_compare = self.__to_pymc_operation(node.left_operation, left, middle)
        right_compare = self.__to_pymc_operation(node.right_operation, middle, right)

        return pm_math.and_(left_compare, right_compare)

    def __handle_index(self, node: Index):
        (operand, _) = self.program_variables[node.operand]
        if isinstance(node.index, str):
            (index, _) = self.program_variables[node.index]
            return operand[index]

        return operand[node.index]

    def __handle_binop(self, node: BinOp):
        left = self.to_pymc(node.left)
        right = self.to_pymc(node.right)
        if isinstance(left, tuple):
            left = left[0]

        if isinstance(right, tuple):
            right = right[0]

        return self.__to_pymc_operation(node.operation, left, right)

    def __handle_attribute(self, node: Attribute):
        (operand, size) = self.to_pymc(node.operand)
        if node.attribute == Operation.SIZE:
            return size

        # TODO: Handle attributes that are not 'sum' and 'size'
        return self.__to_pymc_operation(node.attribute, operand)

    def __handle_call(self, node: Call):
        if isinstance(node.operand, Attribute):
            return self.to_pymc(node.operand)

        mapped_arguments = list(map(self.to_pymc, node.arguments))

        if isinstance(node.operand, Reference):
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
                        return self.to_pymc(child_node, None, True)

                    self.to_pymc(child_node, None, True)

            # TODO: Will it always be a reference to a function?
            print(node.operand.reference_to)
            raise TypeError("Reference to unknown function")

        print(type(node.operand))
        raise TypeError("Unsupported call operand")

    def __handle_if(self, node: If):
        condition = self.to_pymc(node.condition)
        for child_node in node.body:
            self.to_pymc(child_node, condition)

    # Default values for variables declared inside if
    # Used when the condition is not true
    # TODO:
    # Attribute/Call etc will always be the default value no matter the operation?
    # Handle reference to other variable
    # Handle list (other data structures?)
    def __get_default_value(self, node):
        if isinstance(node, Constant):
            if isinstance(node.value, int):
                return (tt.as_tensor_variable(0), None)

            if isinstance(node.value, float):
                return (tt.as_tensor_variable(0.0), None)

        if isinstance(node, ListNode):
            if len(node.values) == 0:
                return ([], 0)

            return (
                tt.as_tensor_variable([tt.as_tensor_variable(0)]) * len(node.values),
                len(node.values),
            )

        if isinstance(node, Subscript):
            (operand, _) = self.program_variables[node.operand]
            lower = self.to_pymc(node.lower) if node.lower else 0
            upper = self.to_pymc(node.upper) if node.upper else len(operand)

            # TODO: Check if continuous or discrete?
            return (
                tt.as_tensor_variable([tt.as_tensor_variable(0)] * (upper - lower)),
                upper - lower,
            )

        # Attribute/Call etc will always be the default value no matter the operation
        # Distinguish between float and int?
        return (tt.as_tensor_variable(0), None)

    def __handle_loop(self, node: Loop):
        start = self.to_pymc(node.start)
        stop = self.to_pymc(node.stop)

        for i in range(start, stop):
            self.program_variables["i"] = (i, None)
            for child_node in node.body:
                self.to_pymc(child_node)

        # Below is kept for future reference.
        # TODO: Verify that this works as expected
        """  outputs, updates = scan(fn=loop_body, sequences=tt.arange(start, stop))
            def loop_body(i, *args):
                self.model_variables['i'] = (i, None)
                for child_node in node.body:
                    self.__map_to_pycm_var(child_node)
                return [] """

        # If necessary, handle the outputs and updates here.
        # For example, you could create a Deterministic variable to store the outputs.
        # pm.Deterministic('loop_outputs', outputs)

        # return outputs, updates

    def __to_pymc_operation(self, operation: Operation, operand, right=None):
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
