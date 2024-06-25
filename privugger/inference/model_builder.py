import theano.tensor as tt
import pymc3 as pm
import pymc3.math as pm_math
import privugger.transformer.PyMC3.ast_types as ast_types
from typing import List
import numpy as np


# Make methods and fields private
class ModelBuilder():
    model_variables = []
    custom_nodes: List[ast_types.CustomNode] = []
    program_arguments = []
    global_priors = []
    num_elements = 0
    program = None
    prior = None
    
    def __init__(self, custom_nodes, program_arguments, global_priors, num_elements, prior, program):
        self.program_arguments = program_arguments
        self.global_priors = global_priors
        self.num_elements = num_elements
        self.custom_nodes = custom_nodes
        self.program = program
        self.prior = prior
        
    def build(self):
        # Map top level function args to PyMC. Args must be of type 'pv.Distribution'
        # TODO: Fix this, arg is distribution and should not be 'concrete' list
        for index, arg_name in enumerate(self.program_arguments):
            self.model_variables.append((arg_name, self.global_priors[index]))
            
        for node in self.custom_nodes:
            self.__map_to_pycm_var(node)

    def __map_to_pycm_var(self, node: ast_types.CustomNode):
        if isinstance(node, ast_types.Subscript):
            return self.__handle_subscript(node)
        
        if isinstance(node, ast_types.Constant):
            return node.value 
        
        if isinstance(node, ast_types.Compare) or isinstance(node, ast_types.Compare2):
            return self.__handle_compare(node)
        
        if isinstance(node, ast_types.Index):
            return self.__handle_index(node)
            
        if isinstance(node, ast_types.BinOp):
            return self.__handle_binop(node)
        
        if isinstance(node, ast_types.Reference):
            return self.__find_dependency(node.reference_to) 
        
        if isinstance(node, ast_types.Assign) or isinstance(node, ast_types.Return): 
            variable = self.__map_to_pycm_var(node.value)
            pymc_var = pm.Deterministic(node.name_with_line_number, tt.as_tensor_variable(variable))
            self.model_variables.append((node.name, pymc_var))
            
            return
        
        if isinstance(node, ast_types.Attribute):
            return self.__handle_attribute(node)
        
        if isinstance(node, ast_types.Call):
            return self.__handle_call(node)
        
        if isinstance(node, ast_types.Loop):
            return self.__handle_loop(node)
        
        if isinstance(node, ast_types.If):
            return self.__handle_if(node)
        
        if isinstance(node, ast_types.ListNode):
            return list(map(lambda x: self.__map_to_pycm_var(x), node.values))
        
        if isinstance(node, ast_types.Distribution):
            if isinstance(node, ast_types.Laplace):
                loc = self.__map_to_pycm_var(node.loc)
                scale = self.__map_to_pycm_var(node.scale)
                return pm.Laplace(node.name_with_line_number, loc, scale)
                    
        print(node)
        raise TypeError("Unsupported custom ast type")
        
    def __handle_subscript(self, node: ast_types.Subscript):
        var = self.__find_dependency(node.dependency_name)
        lower = self.__map_to_pycm_var(node.lower) if node.lower else 0
        upper = self.__map_to_pycm_var(node.upper) if node.upper else len(var)
        if node.dependency_name in self.program_arguments:
            return var[np.arange(self.num_elements)][upper:lower]
        
        return var[lower:upper]
    
    def __handle_compare(self, node):
        left = self.__map_to_pycm_var(node.left)
        right = self.__map_to_pycm_var(node.right)
        if isinstance(node, ast_types.Compare):    
            return self.__to_pymc_operation(node.operation, left, right)
        
        middle = self.__map_to_pycm_var(node.middle)
        left_compare = self.__to_pymc_operation(node.left_operation, left, middle)
        right_compare = self.__to_pymc_operation(node.right_operation, middle, right)
           
        return pm_math.and_(left_compare, right_compare)
    
    def __handle_index(self, node):
        operand = self.__find_dependency(node.dependency_name)
        if isinstance(node.index, str):
            index = self.__find_dependency(node.index)
            return operand[index]
            
        return operand[node.index]
    
    def __handle_binop(self, node: ast_types.BinOp):
        left = self.__map_to_pycm_var(node.left)
        right = self.__map_to_pycm_var(node.right)
        
        return self.__to_pymc_operation(node.operation, left, right)
    
    def __handle_attribute(self, node):
        operand = self.__map_to_pycm_var(node.operand)
        
        # TODO: Handle attributes that are not 'sum' and 'size'
        return self.__to_pymc_operation(node.attribute, operand) 
            
    def __handle_call(self, node):
        if isinstance(node.operand, ast_types.Attribute):
            return self.__map_to_pycm_var(node.operand)  
        
        raise TypeError("Unsupported call operand")
    
    def __handle_if(self, node: ast_types.If):
        condition = self.__map_to_pycm_var(node.condition)
        for index, child_node in enumerate(node.body):
            self.__map_to_pycm_var(child_node)
            
            # TODO: Below is kept for future reference       
            #if isinstance(child_node, ast_types.Assign):
                #print("IN IF ASSIGN")
                # TODO: Does this work as expected?
                # Ask Raul: Should the variable be none if condition is not true?    
                
                
                
                
                #value = self.__map_to_pycm_var(child_node.value, True)
                #switch = pm_math.switch(condition, value, pm.Data(f"D - {child_node.name_with_line_number}", None))
                #pm.Deterministic(child_node.name_with_line_number, switch)
                
            
            #if isinstance(node.type, str):
                #self.__map_to_pycm_var(node.name, node, True)
            
            #self.__map_to_pycm_var(node, True)
                    
            # TODO: The below seems weird. Make sure it's needed
            # Index can be a variable like 'index/i' in a for loop.
            #operand = self.__map_to_pycm_var(child_node.value, True)
            
            #index = self.__find_dependency(child_node.type.index) if isinstance(child_node.type.index, str) else child_node.type.index
            #operand[index] = pm_math.switch(condition, node.value, operand[index])
                
          
    def __handle_loop(self, node):
        start = stop = 0
        if isinstance(node.condition, ast_types.Range):
            start = node.condition.start
            stop = node.condition.stop

        elif isinstance(node.condition, ast_types.OperationOnVariable):
            # Need to handle loops where both start and stop depend on program variables.
            # Variables referenced in loops can't be PyMC variables since they don't work in loops.
            if node.condition.dependency_name in self.program_arguments:
                if node.condition.operation == ast_types.Operation.SIZE:
                    # Assumes input program has only one parameter
                    stop = self.num_elements
            else:
                dependency = self.__find_dependency(node.condition.dependency_name)
                if node.condition.operation == ast_types.Operation.SIZE:
                    stop = len(dependency)
                
                else:
                    raise TypeError("Unsupported loop")
        
        for i in range(start, stop):
            self.model_variables.append(("i", i))
            [self.__map_to_pycm_var(var_name_and_lineno[0], node) for var_name_and_lineno, node in node.body.items()]            
            self.model_variables.pop()

    def __find_dependency(self, dependency_name):
        var_from_model = next((x for x in reversed(self.model_variables) if x[0] == dependency_name), None)
        if var_from_model != None:
            return var_from_model[1]
        
        print(self.model_variables)
        raise TypeError(f'{dependency_name} not found in model_variables')
        
    def __to_pymc_operation(self, operation, operand, right=None):
        if operation == ast_types.Operation.EQUAL:
            return pm_math.eq(operand, right)
        
        if operation == ast_types.Operation.LT:
            return pm_math.lt(operand, right)
        
        if operation == ast_types.Operation.LTE:
            return pm_math.le(operand, right)
        
        if operation == ast_types.Operation.GT:
            return pm_math.gt(operand, right) 
        
        if operation == ast_types.Operation.GTE:
            return pm_math.ge(operand, right) 
        
        if operation == ast_types.Operation.DIVIDE:
            return operand / right
        
        if operation == ast_types.Operation.ADD:
            return operand + right
        
        if operation == ast_types.Operation.SUM:
            print("YES")
            print(operand)
            return pm_math.sum(operand) if isinstance(operand, tt.TensorVariable) else sum(operand)
        
        if operation == ast_types.Operation.SIZE:    
            return operand.shape if isinstance(operand, tt.TensorVariable) else len(operand)
        
        print(operation)
        raise TypeError("Unsupported operation")