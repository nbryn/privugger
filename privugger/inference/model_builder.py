import theano.tensor as tt
import pymc3 as pm
import pymc3.math as pm_math
import privugger.transformer.PyMC3.ast_types as ast_types

# Make methods and fields private
class ModelBuilder():
    dependency_map = {}
    program_params = []
    model_vars = []
    global_priors = []
    num_elements = 0
    program = None
    prior = None
    
    def __init__(self, dependency_map, program_params, global_priors, num_elements, prior, program):
        self.dependency_map = dependency_map
        self.program_params = program_params
        self.global_priors = global_priors
        self.num_elements = num_elements
        self.prior = prior
        self.program = program
    
    def build(self):
        for var_name_and_lineno, node in self.dependency_map.items():
            self.__create_pycm_var(var_name_and_lineno, node)
    
    def __create_pycm_var(self, var_name_and_lineno, node):
        (var_name, line_number) = var_name_and_lineno
        var_name_with_line_number = f"{var_name} - {str(line_number)}"

        var = self.__map_to_pycm_var(var_name, node)
        if not isinstance(node, ast_types.If) and not isinstance(node, ast_types.Loop) and var != None:
            
            # This need to be handled differently
            # Determine wether pymc var or normal var during transformation
            if isinstance(node, ast_types.Assign):
                pymc_var = [0] * self.num_elements 
            else:
                pymc_var = pm.Deterministic(var_name_with_line_number, var if not isinstance(var, tt.TensorVariable) else tt.as_tensor_variable(var))
            
            self.model_vars.append((var_name, pymc_var))

    def __map_to_pycm_var(self, var_name, node, add_to_model_vars=False):
        def helper(func, *args):
            var = func(*args)
            if add_to_model_vars:
                self.model_vars.append((var_name, var))
            
            return var
        
        if isinstance(node, ast_types.Subscript):
            return helper(self.__handle_subscript, node)
        
        elif isinstance(node, ast_types.Constant):
            return node.value 
        
        elif isinstance(node, ast_types.Compare) or isinstance(node, ast_types.Compare2):
            return helper(self.__handle_compare, var_name, node)
        
        elif isinstance(node, ast_types.Index):
            return helper(self.__handle_index, node)
            
        elif isinstance(node, ast_types.BinOp):
            return helper(self.__handle_binop, node)
        
        elif isinstance(node, ast_types.Assign):
            # This needs cleanup. 
            # We should create a 'normal' array when assigning something to the input (look at masking)
            # And create deterministic pymc variable after all mutation
            if isinstance(node.temp, str):
                dependency = self.__find_dependency(node.temp)
                if node.temp == "ages":
                    return dependency
                
                pm.Deterministic(var_name, tt.as_tensor_variable(dependency)) 
                return None
        
        elif isinstance(node, ast_types.Call):
            return helper(self.__handle_call, var_name, node)
        
        elif isinstance(node, ast_types.Loop):
            return helper(self.__handle_loop, var_name, node)
        
        elif isinstance(node, ast_types.If):
            return self.__handle_if(var_name, node)
        
        print(node)
        raise TypeError("Unsupported custom ast type")
        
    def __handle_subscript(self, node):
        var = self.__find_dependency(node.dependency_name)
        return var[node.lower:node.upper]
        #prog.execute_observations(prior, temp) 
    
    def __handle_compare(self, var_name, node):
        left = self.__map_to_pycm_var(var_name, node.left)
        right = self.__map_to_pycm_var(var_name, node.right)
        if isinstance(node, ast_types.Compare):    
            return self.__to_pymc_operation(node.operation, left, right)
        
        middle = self.__map_to_pycm_var(var_name, node.middle)
        left_compare = self.__to_pymc_operation(node.left_operation, left, middle)
        right_compare = self.__to_pymc_operation(node.right_operation, middle, right)
        
        # Does this work as expected?    
        return pm_math.and_(left_compare, right_compare)
    
    def __handle_index(self, node):
        operand = self.__find_dependency(node.dependency_name)
        if isinstance(node.index, str):
            index = self.__find_dependency(node.index)
            return operand[index]
            
        return operand[node.index]
    
    def __handle_binop(self, node):
        left = self.__to_pymc_operation(node.left.operation, self.__find_dependency(node.left.dependency_name))
        right = self.__to_pymc_operation(node.right.operation, self.__find_dependency(node.right.dependency_name))
        
        return self.__to_pymc_operation(node.operation, left, right)
        
        """ if line_number == 17:
            self.prog.execute_observations(self.prior, t) """
            
    def __handle_call(self, var_name, node):
        operand = self.__map_to_pycm_var(var_name, node.operand)
        return self.__to_pymc_operation(node.operation, operand) 
    
    def __handle_if(self, var_name, node):
        body = list(node.body.items())   
        test = self.__map_to_pycm_var(var_name, node.test)
        #body_mapped = [self.__map_to_pycm_var(var_name_and_lineno[0], node, True) for var_name_and_lineno, node in body]
        for index, (var_name_and_lineno, node) in enumerate(body):
            if isinstance(node, ast_types.Assign):
                operand = self.__find_dependency(node.temp.dependency_name)
                index = self.__find_dependency(node.temp.index) if isinstance(node.temp.index, str) else node.temp.index
                val = pm_math.switch(test, node.value, operand[index])
                operand[index] = val
            
            else:    
                var_name_with_line_number = f"{body[index][0]} - {str(body[index][1])}"
                var = self.__map_to_pycm_var(var_name_and_lineno[0], node, True)
                val = pm_math.switch(test, var, pm.Data(f"D - {var_name_with_line_number}", None))
        
                pm.Deterministic(var_name_with_line_number, val)
          
    def __handle_loop(self, var_name, node):
        if isinstance(node.condition, ast_types.Variable):
            start = 0
            stop = 0
            if isinstance(node.condition, ast_types.Range):
                start = node.condition.start
                stop = node.condition.stop

            elif node.condition.dependency_name in self.program_params:
                    stop = self.num_elements
            
            for i in range(start, stop):
                self.model_vars.append(("i", i))
                [self.__map_to_pycm_var(var_name_and_lineno[0], node) for var_name_and_lineno, node in node.body.items()]            
                self.model_vars.pop()
            
        return

    def __find_dependency(self, dependency_name):
        if dependency_name in self.program_params:
            return self.global_priors[0]

        var_from_model = next((x for x in reversed(self.model_vars) if x[0] == dependency_name), None)
        if var_from_model != None:
            return var_from_model[1]
        
        raise TypeError("Couldn't find var in model or program parameters")
        
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
            return pm_math.sum(operand) if isinstance(operand, tt.TensorVariable) else sum(operand)
        
        if operation == ast_types.Operation.SIZE:    
            return operand.shape if isinstance(operand, tt.TensorVariable) else len(operand)
        
        print(operation)
        raise TypeError("Unsupported operation")
        
        
        
    
    
    
        
            