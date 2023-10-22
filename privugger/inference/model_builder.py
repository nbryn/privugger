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
            self.create_pycm_var(var_name_and_lineno, node)
    
    def create_pycm_var(self, var_name_and_lineno, node):
        (var_name, line_number) = var_name_and_lineno
        var_name_with_line_number = f"{var_name} - {str(line_number)}"

        var = self.map(var_name, node)
        pymc_var = pm.Deterministic(var_name_with_line_number, tt.as_tensor_variable(var))
        self.model_vars.append((var_name, pymc_var))

    def map(self, var_name, node):
        if isinstance(node, ast_types.Subscript):
            return self.handle_subscript(var_name, node)
        
        elif isinstance(node, ast_types.Constant):
            return node.value 
        
        elif isinstance(node, ast_types.Compare):
            return self.handle_compare(var_name, node)
        
        elif isinstance(node, ast_types.Index):
            return self.handle_index(node)                      
            
        elif isinstance(node, ast_types.BinOp):
            return self.handle_binop(node)
        
        elif isinstance(node, ast_types.Call):
            return self.handle_call(var_name, node)
        
        elif isinstance(node, ast_types.If):
            return self.handle_if(var_name, node)
        
        raise TypeError("Unsupported custom ast type")
        
    def handle_subscript(self, var_name, node):
        var = None
        if node.dependency_name in self.program_params:
            var = self.global_priors[0]
        elif node.dependency_name in self.model_vars:
            var = next(x for x in self.model_vars.reverse() if var_name == x)
        
        # else: Need to look up value of variable in ast tree (constant etc)                    
        
        # Does array slicing work if var is a distribution?
        # Should work the same as numpy indexing
        return var[node.lower:node.upper]
        #prog.execute_observations(prior, temp) 
    
    def handle_compare(self, var_name, node):
        left = self.map(var_name, node.left)
        right = self.map(var_name, node.right)
        if node.operation == ast_types.Operation.EQUAL:
            return pm_math.eq(left, right)
        
        if node.operation == ast_types.Operation.LT:
            return pm_math.lt(left, right)
        
        if node.operation == ast_types.Operation.GT:
            return pm_math.gt(left, right) 
    
    def handle_index(self, node):
        operand = None
        var_from_model = next((x for x in reversed(self.model_vars) if x[0] == node.dependency_name), None)
        if node.dependency_name in self.program_params:
            operand = self.global_priors[0]
        elif var_from_model != None:
            operand = var_from_model[1]
        
        return operand[node.index]
    
    def handle_binop(self, node):
        left = None
        var_from_model = next((x for x in reversed(self.model_vars) if x[0] == node.left.dependency_name), None)
        if node.left.dependency_name in self.program_params:
            left = self.global_priors[0]
        elif var_from_model != None:
            left = var_from_model[1]
        
        # else: Need to look up value of variable in ast tree (constant etc)
        
        if node.left.operation == ast_types.Operation.SUM:
            left = pm_math.sum(left) if isinstance(left, tt.TensorVariable) else sum(left)
        elif node.right.operation == ast_types.Operation.SIZE:    
            left = left.shape if isinstance(left, tt.TensorVariable) else len(left)
        
        right = None
        var_from_model = next((x for x in reversed(self.model_vars) if x[0] == node.right.dependency_name), None)
        if node.right.dependency_name in self.program_params:
            right = self.global_priors[0]
        elif var_from_model != None:
            right = var_from_model[1]
        
        # else: Need to look up value of variable in ast tree (constant etc)
        
        if node.right.operation == ast_types.Operation.SUM:
            right = pm_math.sum(right) if isinstance(right, tt.TensorVariable) else sum(right)
        elif node.right.operation == ast_types.Operation.SIZE:    
            right = right.shape if isinstance(right, tt.TensorVariable) else len(right)
        
        if node.operation == ast_types.Operation.DIVIDE:
            return left / right
        elif node.operation == ast_types.Operation.ADD:
            return left + right
        
        """ if line_number == 17:
            self.prog.execute_observations(self.prior, t) """
    
    def handle_call(self, var_name, node):
        operand = self.map(var_name, node.operand)
        if node.operation == ast_types.Operation.SUM:
            return pm_math.sum(operand) if isinstance(operand, tt.TensorVariable) else sum(operand)
        elif node.operation == ast_types.Operation.SIZE:    
            return operand.shape if isinstance(operand, tt.TensorVariable) else len(operand)
    
    # Cant use 'if <boolean>' so need to evaluate whole body of if?
    # That means only measuring risk in the final variable/return of if?
    # Measure risk of whole if branch as a whole?
    # Ask Raul about best way to approach this?
    # - Maybe possible using aesera?
    def handle_if(self, var_name, node):
        print("HERE")
        test = self.map(var_name, node.test)
        body_mapped = [self.map(var_name_and_lineno[0], node) for var_name_and_lineno, node in node.body.items()]
        print(body_mapped)
        """ for var_name_and_lineno, node in node.body.items():
                self.create_pycm_var(var_name_and_lineno, node)
        _ = pm_math.switch(test, self.loop(node), tt.as_tensor_variable(0)) """
        
        """ if node.else_branch:
            print("ELSE")  """
            