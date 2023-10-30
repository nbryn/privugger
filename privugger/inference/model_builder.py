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
    program = None
    prior = None
    
    def __init__(self, dependency_map, program_params, global_priors, prior, program):
        self.dependency_map = dependency_map
        self.program_params = program_params
        self.global_priors = global_priors
        self.prior = prior
        self.program = program
    
    def build(self):
        for var_name_and_lineno, node in self.dependency_map.items():
            self.__create_pycm_var(var_name_and_lineno, node)
    
    def __create_pycm_var(self, var_name_and_lineno, node):
        (var_name, line_number) = var_name_and_lineno
        var_name_with_line_number = f"{var_name} - {str(line_number)}"

        var = self.__map_to_pycm_var(var_name, node)
        if not isinstance(node, ast_types.If):
            pymc_var = pm.Deterministic(var_name_with_line_number, tt.as_tensor_variable(var))
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
        
        elif isinstance(node, ast_types.Compare):
            return helper(self.__handle_compare, var_name, node)
        
        elif isinstance(node, ast_types.Index):
            return helper(self.__handle_index, node)
            
        elif isinstance(node, ast_types.BinOp):
            return helper(self.__handle_binop, node)
        
        elif isinstance(node, ast_types.Call):
            return helper(self.__handle_call, var_name, node)
        
        elif isinstance(node, ast_types.If):
            return self.__handle_if(var_name, node)
        
        raise TypeError("Unsupported custom ast type")
        
    def __handle_subscript(self, node):
        var = self.__temp(node.dependency_name)
        return var[node.lower:node.upper]
        #prog.execute_observations(prior, temp) 
    
    def __handle_compare(self, var_name, node):
        left = self.__map_to_pycm_var(var_name, node.left)
        right = self.__map_to_pycm_var(var_name, node.right)
        return self.__to_pymc_operation(node.operation, left, right)
        
    def __handle_index(self, node):
        operand = self.__temp(node.dependency_name)
        return operand[node.index]
    
    def __handle_binop(self, node):
        left = self.__to_pymc_operation(node.left.operation, self.__temp(node.left.dependency_name))
        right = self.__to_pymc_operation(node.right.operation, self.__temp(node.right.dependency_name))
        
        return self.__to_pymc_operation(node.operation, left, right)
        
        """ if line_number == 17:
            self.prog.execute_observations(self.prior, t) """
    
    def __handle_call(self, var_name, node):
        operand = self.__map_to_pycm_var(var_name, node.operand)
        return self.__to_pymc_operation(node.operation, operand) 
    
    def __handle_if(self, var_name, node):
        last = list(node.body.items())[-1][0]        
        var_name_with_line_number = f"{last[0]} - {str(last[1])}"
        test = self.__map_to_pycm_var(var_name, node.test)
        body_mapped = [self.__map_to_pycm_var(var_name_and_lineno[0], node, True) for var_name_and_lineno, node in node.body.items()]
        
        val = pm_math.switch(test, body_mapped[-1], pm.Data(f"D - {var_name_with_line_number}", None))
        pm.Deterministic(var_name_with_line_number, val)
        
    def __temp(self, dependency_name):
        var_from_model = next((x for x in reversed(self.model_vars) if x[0] == dependency_name), None)
        if dependency_name in self.program_params:
            return self.global_priors[0]
        
        elif var_from_model != None:
            return var_from_model[1]
        
        raise TypeError("Couldn't find var in model or program parameters")
        
    def __to_pymc_operation(self, operation, operand, right=None):
        if operation == ast_types.Operation.EQUAL:
            return pm_math.eq(operand, right)
        
        if operation == ast_types.Operation.LT:
            return pm_math.lt(operand, right)
        
        if operation == ast_types.Operation.GT:
            return pm_math.gt(operand, right) 
        
        if operation == ast_types.Operation.DIVIDE:
            return operand / right
        
        if operation == ast_types.Operation.ADD:
            return operand + right
        
        if operation == ast_types.Operation.SUM:
            return pm_math.sum(operand) if isinstance(operand, tt.TensorVariable) else sum(operand)
        
        if operation == ast_types.Operation.SIZE:    
            return operand.shape if isinstance(operand, tt.TensorVariable) else len(operand)
        
        raise TypeError("Unknown operation")
        
        
        
    
    
    
        
            