import ast
import privugger.transformer.PyMC3.ast_types as ast_types

class Transformer():
    tree = None
    dependency_map = {}
    function_def = None
    used_nodes = set()
    
    def __init__(self, tree, function_def):
        self.tree = tree
        self.function_def = function_def
    
    
    def transform(self):
        function_params = list(map(lambda x: x.arg, self.function_def.args.args))
        relevant_ast_nodes = self.__collect_relevant_ast_nodes(self.tree)
            
        for node in relevant_ast_nodes:
            var_name = self.__get_variable_name(node)
            if var_name in self.used_nodes: continue
            self.__collect_dependencies(node, self.dependency_map)

        return (function_params, dict(sorted(self.dependency_map.items(), key=lambda x: x[0][1])))
    
    
    def __collect_relevant_ast_nodes(self, root):
        nodes = []
        for child_node in ast.walk(root):
            if isinstance(child_node, ast.If):
                nodes.append(child_node)
            
            if isinstance(child_node, ast.Assign):
                nodes.append(child_node)
            
            if isinstance(child_node, ast.Return):
                if not isinstance(child_node.value, ast.Name):                    
                    nodes.append(child_node)

        return sorted(nodes, key=lambda x: x.lineno)
    
    def __collect_dependencies(self, node, dependencies):
        var_name = self.__get_variable_name(node)
        self.used_nodes.add(var_name)
        if isinstance(node, ast.If):
           type = self.__map_ast_if_to_custom_type(node)
           dependencies[var_name] = type
        else:      
            type = self.__map_ast_node_to_custom_type(node)
            dependencies[var_name] = type
    
    def __map_ast_if_to_custom_type(self, node):
        test = self.__map_ast_node_to_custom_type(ast_types.WrapperNode(node.test))
         # This to handle nested if's
        body_dependency_map = {}
        for child_node in node.body:
            # Ignore returns that just return a variable with no additional computation
            if not isinstance(child_node.value, ast.Name): 
                self.__collect_dependencies(child_node, body_dependency_map)

        # Need to handle else here?
        # elif is an ast.If
        
        return ast_types.If(test, dict(sorted(body_dependency_map.items(), key=lambda x: x[0][1])))
    
    def __map_ast_node_to_custom_type(self, node):
        value = node.value
        if isinstance(value, ast.Call):
            operand = self.__map_ast_node_to_custom_type(value.func)
            return ast_types.Call(operand, self.__map_to_operation(value.func.attr))
            
        if isinstance(value, int):
            return ast_types.Constant(value)
        
        if isinstance(value, ast.Compare):
            left = self.__map_ast_node_to_custom_type(ast_types.WrapperNode(value.left))
            # Assumes only one comparator
         
            right = self.__map_ast_node_to_custom_type(value.comparators[0])
            
            # Assumes only one operation
            operation = self.__map_to_operation(value.ops[0])
            return ast_types.Compare(left, right, operation)
            
        
        if isinstance(value, ast.Subscript):
            subscript = ast_types.Subscript()
            subscript.dependency_name = value.value.id
           
            if isinstance(value.slice, ast.Index):
                # Assumes index is a constant
                return ast_types.Index(value.value.id, value.slice.value.value)
            
             # Assumes lower and upper has constant values
            if hasattr(value.slice.lower, "value"):
                subscript.lower = value.slice.lower.value
                
            if hasattr(value.slice.upper, "value"):
                subscript.upper = value.slice.upper.value
            
            return subscript

        if isinstance(value, ast.BinOp):
            binop = ast_types.BinOp()
            binop.operation = self.__map_to_operation(value.op)
            if isinstance(value.left.func, ast.Attribute):
                binop.left = ast_types.Variable(value.left.func.value.id, self.__map_to_operation(value.left.func.attr))
                 
            if isinstance(value.right, ast.Attribute):
                binop.right = ast_types.Variable(value.right.value.id, self.__map_to_operation(value.right.attr))
                
            return binop
        
        raise TypeError("ast type not supported")
    
    def __get_variable_name(self, node):
        # Named variable with name
        # This assumes there is only one target
        if hasattr(node, "targets"):
            return (node.targets[0].id, node.lineno)
        
        if isinstance(node, ast.If):
            return ("if", node.lineno)
        
        if isinstance(node, ast.Return):
            return ("return", node.lineno) 
        
    def __map_to_operation(self, operation):
        if operation == "sum":
            return ast_types.Operation.SUM
        
        if operation == "size":
            return ast_types.Operation.SIZE
        
        if isinstance(operation, ast.Add):
            return ast_types.Operation.ADD    
        
        if isinstance(operation, ast.Div):
            return ast_types.Operation.DIVIDE
        
        if isinstance(operation, ast.Mult):
            return ast_types.Operation.MULTIPLY        
        
        if isinstance(operation, ast.Lt):
            return ast_types.Operation.LT
        
        print(operation)
        raise TypeError("Unknown operation") 
        
        
    