import privugger.white_box.model as model
from collections.abc import Sized
import pymc3.math as pm_math
import theano.tensor as tt
from theano.scan import scan
from typing import List
import pymc3 as pm
import random

# Make methods and fields private
class PyMCModelBuilder:
    custom_nodes: List[model.CustomNode] = []
    program_arguments = []
    model_variables = {}
    global_priors = []
    num_elements = 0
    pymc_model = None

    def __init__(
        self,
        custom_nodes,
        program_arguments,
        global_priors,
        num_elements,
        pymc_model
    ):
        self.program_arguments = program_arguments
        self.global_priors = global_priors
        self.num_elements = num_elements
        self.custom_nodes = custom_nodes
        self.pymc_model = pymc_model

    def build(self):
        # Map top level function args to PyMC. Args must be of type 'pv.Distribution'
        for index, arg_name in enumerate(self.program_arguments):
            print("HEEEEEEEEEEEEEEER")
            # TODO: Can we have more than one argument?
            print(self.num_elements)
            self.model_variables[arg_name] = (
                self.global_priors[index],
                self.num_elements,
            )

        for node in self.custom_nodes:
            self.__map_to_pycm_var(node)

    def __map_to_pycm_var(self, node: model.CustomNode):
        def helper(fn):
            var = fn()
            return var[0] if isinstance(var, tuple) else var
            
        if isinstance(node, model.Subscript):
            return self.__handle_subscript(node)

        if isinstance(node, model.Constant):
            return node.value

        if isinstance(node, model.Compare) or isinstance(node, model.Compare2):
            return self.__handle_compare(node)

        if isinstance(node, model.Index):
            return self.__handle_index(node)

        if isinstance(node, model.BinOp):
            return self.__handle_binop(node)

        if isinstance(node, model.Reference):
            return self.model_variables[node.reference_to]

        if isinstance(node, model.Assign) or isinstance(node, model.Return):
            # TODO: Split into assign and return and create handle_methods
            variable = self.__map_to_pycm_var(node.value)
            if isinstance(variable, tuple):
                variable = variable[0]

            # Handle assignment to parameter/argument
            # We shouldn't (and can't) change the input as it's a distribution
            if isinstance(node.value, model.Reference) and node.value.reference_to in self.program_arguments:
                (operand, _) = self.model_variables[node.value.reference_to]
                variable = operand[:self.num_elements] 

            tensor_var = tt.as_tensor_variable(variable)
            if isinstance(node, model.AssignIndex):
                print("OMG")
                (index, _) = self.__map_to_pycm_var(node.index)
                print(index)
                print(self.model_variables)
                (operand, _) = self.model_variables[node.name]
                tensor_var = tt.set_subtensor(operand[index], tensor_var)
                del self.pymc_model.named_vars[node.name]
            
            # TODO: Does it create problem to only use name without line number?
            #self.model_variables[node.name_with_line_number] = (pymc_var, size)
            size = len(variable) if isinstance(variable, Sized) else None
            self.model_variables[node.name] = (tensor_var, size)
            pm.Deterministic(node.name, tensor_var)
            
            return

        if isinstance(node, model.Attribute):
            return self.__handle_attribute(node)

        if isinstance(node, model.Call):
            return self.__handle_call(node)

        if isinstance(node, model.Loop):
            return self.__handle_loop(node)

        if isinstance(node, model.If):
            return self.__handle_if(node)

        if isinstance(node, model.ListNode):
            return list(map(self.__map_to_pycm_var, node.values))

        if isinstance(node, model.Distribution):
            if isinstance(node, model.Laplace):
                loc = self.__map_to_pycm_var(node.loc)
                scale = self.__map_to_pycm_var(node.scale)
                return pm.Laplace(node.name_with_line_number, loc, scale)

        print(node)
        raise TypeError("Unsupported custom ast type")

    def __handle_subscript(self, node: model.Subscript):
        (operand, _) = self.model_variables[node.operand]
        lower = self.__map_to_pycm_var(node.lower) if node.lower else 0
        upper = self.__map_to_pycm_var(node.upper) if node.upper else len(operand)

        return operand[lower:upper]

    def __handle_compare(self, node):
        left = self.__map_to_pycm_var(node.left)
        right = self.__map_to_pycm_var(node.right)
        if isinstance(node, model.Compare):
            return self.__to_pymc_operation(node.operation, left, right)

        middle = self.__map_to_pycm_var(node.middle)
        left_compare = self.__to_pymc_operation(node.left_operation, left, middle)
        right_compare = self.__to_pymc_operation(node.right_operation, middle, right)

        return pm_math.and_(left_compare, right_compare)

    def __handle_index(self, node: model.Index):
        (operand, _) = self.model_variables[node.operand]
        if isinstance(node.index, str):
            (index, _) = self.model_variables[node.index]
            return operand[index]
        
        return operand[node.index]

    def __handle_binop(self, node: model.BinOp):
        left = self.__map_to_pycm_var(node.left)
        right = self.__map_to_pycm_var(node.right)
        if isinstance(left, tuple):
            left = left[0]
        
        if isinstance(right, tuple):
            right = right[0]

        return self.__to_pymc_operation(node.operation, left, right)

    def __handle_attribute(self, node):
        operand = self.__map_to_pycm_var(node.operand)

        # TODO: Handle attributes that are not 'sum' and 'size'
        return self.__to_pymc_operation(node.attribute, operand)

    def __handle_call(self, node):
        if isinstance(node.operand, model.Attribute):
            return self.__map_to_pycm_var(node.operand)

        raise TypeError("Unsupported call operand")

    def __handle_if(self, node: model.If):
        condition = self.__map_to_pycm_var(node.condition)
        for index, child_node in enumerate(node.body):
            self.__map_to_pycm_var(child_node)

            # TODO: Below is kept for future reference
            # if isinstance(child_node, ast_types.Assign):
            # print("IN IF ASSIGN")
            # TODO: Does this work as expected?
            # Ask Raul: Should the variable be none if condition is not true?

            # value = self.__map_to_pycm_var(child_node.value, True)
            # switch = pm_math.switch(condition, value, pm.Data(f"D - {child_node.name_with_line_number}", None))
            # pm.Deterministic(child_node.name_with_line_number, switch)

            # if isinstance(node.type, str):
            # self.__map_to_pycm_var(node.name, node, True)

            # self.__map_to_pycm_var(node, True)

            # TODO: The below seems weird. Make sure it's needed
            # Index can be a variable like 'index/i' in a for loop.
            # operand = self.__map_to_pycm_var(child_node.value, True)

            # index = self.__find_dependency(child_node.type.index) if isinstance(child_node.type.index, str) else child_node.type.index
            # operand[index] = pm_math.switch(condition, node.value, operand[index])

    def __handle_loop(self, node: model.Loop):
        print("LOOP")
        print(self.model_variables)
        start = self.__map_to_pycm_var(node.start)
        stop = self.__map_to_pycm_var(node.stop)
        print(self.model_variables)
        print(start)
        print(stop)

        for i in range(start, stop+1):
            self.model_variables["i"] = (i, None)
            for child_node in node.body:
                self.__map_to_pycm_var(child_node)


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
        #pm.Deterministic('loop_outputs', outputs)

        #return outputs, updates

    def __to_pymc_operation(self, operation, operand, right=None, return_pymc_var=True):
        if operation == model.Operation.EQUAL:
            return pm_math.eq(operand, right)

        if operation == model.Operation.LT:
            return pm_math.lt(operand, right)

        if operation == model.Operation.LTE:
            return pm_math.le(operand, right)

        if operation == model.Operation.GT:
            return pm_math.gt(operand, right)

        if operation == model.Operation.GTE:
            return pm_math.ge(operand, right)

        if operation == model.Operation.DIVIDE:
            return operand / right

        if operation == model.Operation.ADD:
            return operand + right

        if operation == model.Operation.SUM:
            return (
                pm_math.sum(operand)
                if isinstance(operand, tt.TensorVariable)
                else sum(operand)
            )

        if operation == model.Operation.SIZE:
            return (
                operand.shape[0]
                if isinstance(operand, tt.TensorVariable)
                else len(operand)
            )

        print(operation)
        raise TypeError("Unsupported operation")
