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
        nodes_to_collect = {ast.If, ast.For, ast.Assign, ast.Return}
        nodes = []
        for child_node in ast.walk(root):
            if child_node.__class__ in nodes_to_collect:
                if isinstance(child_node, ast.Return): 
                    # Ignore returns that just return a variable with no additional computation
                    #if not isinstance(child_node.value, ast.Name):                    
                    nodes.append(child_node)
                else:
                    nodes.append(child_node)
                    
        return sorted(nodes, key=lambda x: x.lineno)
    
    def __collect_dependencies(self, node, dependencies):
        var_name = self.__get_variable_name(node)
        self.used_nodes.add(var_name)        
        if isinstance(node, ast.If):
           type = self.__map_ast_if_to_custom_type(node)
           dependencies[var_name] = type
        elif isinstance(node, ast.For):
            type = self.__map_ast_for_to_custom_type(node)
            dependencies[var_name] = type
        else:      
            type = self.__map_ast_node_to_custom_type(node)
            dependencies[var_name] = type
    
    def __map_ast_if_to_custom_type(self, node):
        test = self.__map_ast_node_to_custom_type(ast_types.WrapperNode(node.test))
         # This to handle nested if's
        body_dependency_map = self.__handle_if_or_loop_body(node)
        # Need to handle else here?
        # elif is an ast.If
        
        return ast_types.If(test, body_dependency_map)
    
    # Loops are not well supported in PyMC.
    # So loops shouldn't be translated into PyMC variables
    # but instead function as a 'normal' python loop.
    def __map_ast_for_to_custom_type(self, node):
        # Only standard 'for i in range()' looprs supported for now
        body_dependency_map = self.__handle_if_or_loop_body(node)
        args = node.iter.args
        if isinstance(args[0], ast.Call):
            if isinstance(args[0].args[0], ast.Constant):
                range = ast_types.Range(args[0].args[0].value)
                return ast_types.Loop(range, body_dependency_map)
            
            elif isinstance(args[0].args[0], ast.Name):            
                operation = self.__map_to_operation(args[0].func.id)
                dependency = args[0].args[0].id
                var = ast_types.Variable(dependency, operation)
            
                return ast_types.Loop(var, body_dependency_map)
        
        raise TypeError("Unsuported node encountered in for loop")
    
    def __handle_if_or_loop_body(self, node):        
        body_dependency_map = {}
        for child_node in node.body:
            self.__collect_dependencies(child_node, body_dependency_map)
        
        return dict(sorted(body_dependency_map.items(), key=lambda x: x[0][1]))
    
    def __map_ast_node_to_custom_type(self, node):
        value = node.value if (not isinstance(node, ast.Assign) and not isinstance(node, ast.Subscript) and not isinstance(node, ast.Return)) else node
        if isinstance(value, ast.Call):
            operand = self.__map_ast_node_to_custom_type(value.func)
            return ast_types.Call(operand, self.__map_to_operation(value.func.attr))
            
        if isinstance(value, ast.Constant) or isinstance(value, int):
            return ast_types.Constant(value if isinstance(value, int) else value.value)
        
        if isinstance(value, ast.Compare):
            return self.__handle_compare(value)
            
        if isinstance(value, ast.Subscript):
            return self.__handle_subscript(value)

        if isinstance(value, ast.BinOp):
            return self.__handle_binop(value)

        if isinstance(value, ast.Assign):
            # Assumes only one target
            if isinstance(value.value, ast.Name):
                return ast_types.Assign(value.value.id)
             
            temp = self.__map_ast_node_to_custom_type(value.targets[0])
            value = value.value.value
            return ast_types.Assign(temp, value)

        if isinstance(value, ast.Return):
            return ast_types.Assign(value.value.id)

        print(ast.dump(value))
        raise TypeError("ast type not supported")
    
    def __handle_compare(self, value):
        left = self.__map_ast_node_to_custom_type(ast_types.WrapperNode(value.left))
        # Assumes max two comparators
        left_operation = self.__map_to_operation(value.ops[0])
        middle_or_right = self.__map_ast_node_to_custom_type(value.comparators[0])   
        if len(value.comparators) < 2:
            return ast_types.Compare(left, middle_or_right, left_operation)
        
        right_operation = self.__map_to_operation(value.ops[1])
        right = self.__map_ast_node_to_custom_type(value.comparators[1])
            
        return ast_types.Compare2(left, left_operation, middle_or_right, right, right_operation)
    
    def __handle_subscript(self, value):
        subscript = ast_types.Subscript()
        subscript.dependency_name = value.value.id
        
        if isinstance(value.slice, ast.Index):
            return ast_types.Index(value.value.id, value.slice.value.value if hasattr(value.slice.value, 'value') else value.slice.value.id)
        
        # Assumes lower and upper has constant values
        if hasattr(value.slice.lower, "value"):
            subscript.lower = value.slice.lower.value
            
        if hasattr(value.slice.upper, "value"):
            subscript.upper = value.slice.upper.value
        
        return subscript
    
    def __handle_binop(self, value):
        binop = ast_types.BinOp()
        binop.operation = self.__map_to_operation(value.op)
        if isinstance(value.left.func, ast.Attribute):
            binop.left = ast_types.Variable(value.left.func.value.id, self.__map_to_operation(value.left.func.attr))
                
        if isinstance(value.right, ast.Attribute):
            binop.right = ast_types.Variable(value.right.value.id, self.__map_to_operation(value.right.attr))
            
        return binop
    
    def __get_variable_name(self, node):
        # Named variable with name
        # This assumes there is only one target
        if hasattr(node, "targets"):
            if hasattr(node.targets[0], "id"):
                return (node.targets[0].id, node.lineno)

            return (node.targets[0].value.id, node.lineno)
        
        if isinstance(node, ast.If):
            return ("if", node.lineno)
        
        if isinstance(node, ast.For):
            return ("for", node.lineno)
        
        if isinstance(node, ast.Return):
            return ("return", node.lineno) 
        
        print(ast.dump(node))
        raise TypeError("Unsupported")
        
    def __map_to_operation(self, operation):
        if operation == "sum":
            return ast_types.Operation.SUM
        
        if operation == "size" or operation == "len":
            return ast_types.Operation.SIZE
        
        if isinstance(operation, ast.Add):
            return ast_types.Operation.ADD    
        
        if isinstance(operation, ast.Div):
            return ast_types.Operation.DIVIDE
        
        if isinstance(operation, ast.Mult):
            return ast_types.Operation.MULTIPLY        
        
        if isinstance(operation, ast.Lt):
            return ast_types.Operation.LT
        
        if isinstance(operation, ast.LtE):
            return ast_types.Operation.LTE
        
        if isinstance(operation, ast.Gt):
            return ast_types.Operation.GT
        
        if isinstance(operation, ast.GtE):
            return ast_types.Operation.GTE
        
        print(operation)
        raise TypeError("Unknown operation") 
        
        
    