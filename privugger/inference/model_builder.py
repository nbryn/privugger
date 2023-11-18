import theano.tensor as tt
import pymc3 as pm
import pymc3.math as pm_math
import privugger.transformer.PyMC3.ast_types as ast_types

# Make methods and fields private
class ModelBuilder():
    model_variables = []
    dependency_map = {}
    program_params = []
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
        
        # Handles top level assignment that references/points directly to parameter.
        # Will fail if assignment isn't a direct reference
        if isinstance(node, ast_types.Assign) and node.temp in self.program_params:
            # Assumes single parameter that is an array/distribution.
            # Better way to initialize here?
            var = self.global_priors[0].random().tolist()            
            self.model_variables.append((var_name, var))
            return
        
        if isinstance(node, ast_types.Laplace):
            loc = self.__map_to_pycm_var(var_name, node.loc)
            scale = self.__map_to_pycm_var(var_name, node.scale)
            var = pm.Laplace(var_name_with_line_number, loc, scale)
            self.model_variables.append((var_name, var))
            return 
            
        var = self.__map_to_pycm_var(var_name, node)
        if not isinstance(node, ast_types.If) and not isinstance(node, ast_types.Loop):
            pymc_var = pm.Deterministic(var_name_with_line_number, tt.as_tensor_variable(var))
            self.model_variables.append((var_name, pymc_var))

    def __map_to_pycm_var(self, var_name, node, add_to_model_vars=False):
        def helper(func, *args):
            var = func(*args)
            if add_to_model_vars:
                self.model_variables.append((var_name, var))
            
            return var
        
        if isinstance(node, ast_types.Subscript):
            return helper(self.__handle_subscript, node)
        
        if isinstance(node, ast_types.Constant):
            return node.value 
        
        if isinstance(node, ast_types.Compare) or isinstance(node, ast_types.Compare2):
            return helper(self.__handle_compare, var_name, node)
        
        if isinstance(node, ast_types.Index):
            return helper(self.__handle_index, node)
            
        if isinstance(node, ast_types.BinOp):
            return helper(self.__handle_binop, node)
        
        if isinstance(node, ast_types.Assign):
            if isinstance(node.temp, str):
                return self.__find_dependency(node.temp)
            
            if isinstance(node.temp, ast_types.Constant):
                return node.temp.value
            
            if isinstance(node.temp, ast_types.Index):
                dependency = self.__find_dependency(node.temp.dependency_name)
                return dependency[node.temp.index]
                
            raise TypeError("Unsupported assign operation")
        
        if isinstance(node, ast_types.Call):
            return helper(self.__handle_call, var_name, node)
        
        if isinstance(node, ast_types.Loop):
            return helper(self.__handle_loop, node)
        
        if isinstance(node, ast_types.If):
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
           
        return pm_math.and_(left_compare, right_compare)
    
    def __handle_index(self, node):
        operand = self.__find_dependency(node.dependency_name)
        if isinstance(node.index, str):
            index = self.__find_dependency(node.index)
            return operand[index]
            
        return operand[node.index]
    
    def __handle_binop(self, node):
        if isinstance(node.left, ast_types.Reference):
            left = self.__find_dependency(node.left.dependency_name)    
        elif isinstance(node.left, ast_types.Constant):
            left = node.left.value
        else:        
            left = self.__to_pymc_operation(node.left.operation, self.__find_dependency(node.left.dependency_name))
        
        if isinstance(node.right, ast_types.Reference):
            right = self.__find_dependency(node.right.dependency_name)
        elif isinstance(node.right, ast_types.Constant):
            right = node.right.value
        else:
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
        for index, (var_name_and_lineno, node) in enumerate(body):
            if isinstance(node, ast_types.Assign):
                if isinstance(node.temp, ast_types.Index):
                    operand = self.__find_dependency(node.temp.dependency_name)
                    # Index can be a variable like 'index/i' in a for loop.
                    index = self.__find_dependency(node.temp.index) if isinstance(node.temp.index, str) else node.temp.index
                    operand[index] = pm_math.switch(test, node.value, operand[index])
                
                elif isinstance(node.temp, str):
                    self.__map_to_pycm_var(var_name_and_lineno[0], node, True)
                
                else:
                    self.__map_to_pycm_var(var_name_and_lineno[0], node.temp, True)
                    
            else:    
                var_name_with_line_number = f"{body[index][0]} - {str(body[index][1])}"
                var = self.__map_to_pycm_var(var_name_and_lineno[0], node, True)
                val = pm_math.switch(test, var, pm.Data(f"D - {var_name_with_line_number}", None))
        
                pm.Deterministic(var_name_with_line_number, val)
          
    def __handle_loop(self, node):
        start = stop = 0
        if isinstance(node.condition, ast_types.Range):
            start = node.condition.start
            stop = node.condition.stop

        elif isinstance(node.condition, ast_types.Variable):
            # Need to handle loops where both start and stop depend on program variables.
            # Variables referenced in loops can't be PyMC variables since they don't work in loops.
            if node.condition.dependency_name in self.program_params:
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
        if dependency_name in self.program_params:
            return self.global_priors[0]

        var_from_model = next((x for x in reversed(self.model_variables) if x[0] == dependency_name), None)
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
        
        
        
    
    
    
        
            