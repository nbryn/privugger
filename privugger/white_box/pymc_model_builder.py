from .transformer_factory import TransformerFactory
from .transformers import *
from collections.abc import Sized
import pymc3.math as pm_math
import theano.tensor as tt
from typing import List
import pymc3 as pm


# Make methods and fields private
# TODO: Change name to ModelBuilder and pass the type of model to build
# Transformer are responsible of constructing model while this class holds state
class ModelBuilder:
    transformer_factory = TransformerFactory()
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
        
        return self.transformer_factory.create_transformer(node).to_pymc(node, self, condition, in_function)

    
    # Default values for variables declared inside if
    # Used when the condition is not true
    # TODO:
    # Attribute/Call etc will always be the default value no matter the operation?
    # Handle reference to other variable
    # Handle list (other data structures?)
    def get_default_value(self, node):
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

    def to_pymc_operation(self, operation: Operation, operand, right=None):
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
