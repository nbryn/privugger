import ast
import privugger.transformer.PyMC3.ast_types as ast_types

class Transformer():
    tree = None
    variable_name_to_custom_type_map = {}
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
            self.__collect_dependencies(node)

        return (function_params, dict(sorted(self.variable_name_to_custom_type_map.items(), key=lambda x: x[0][1])))
    
    def __collect_relevant_ast_nodes(self, root):
        nodes_to_collect = {ast.If, ast.For, ast.Assign, ast.Return}
        nodes = []
        for child_node in ast.walk(root):
            if child_node.__class__ in nodes_to_collect:
                nodes.append(child_node)
                    
        return sorted(nodes, key=lambda x: x.lineno)
    
    def __collect_dependencies(self, node):
        variable_name = self.__get_variable_name(node)
        self.used_nodes.add(variable_name)
        custom_type = None        
        if isinstance(node, ast.If):
           custom_type = self.__map_if_to_custom_type(node)
        elif isinstance(node, ast.For):
            custom_type = self.__map_for_to_custom_type(node)
        else:      
            custom_type = self.__map_node_to_custom_type(node)
        
        self.variable_name_to_custom_type_map[variable_name] = custom_type
    
    def __map_if_to_custom_type(self, node):
        test = self.__map_node_to_custom_type(node.test)
         # This to handle nested if's
        body_dependency_map = self.__handle_if_or_loop_body(node)
        # Need to handle else here?
        # elif is an ast.If
        
        return ast_types.If(test, body_dependency_map)
    
    # Loops are not well supported in PyMC.
    # So loops shouldn't be translated into PyMC variables
    # but instead function as a 'normal' python loop.
    def __map_for_to_custom_type(self, node):
        # Only standard 'for i in range()' loops supported for now
        body_dependency_map = self.__handle_if_or_loop_body(node)
        args = node.iter.args
        if isinstance(args[0], ast.Call):
            if isinstance(args[0].args[0], ast.Constant):
                range = ast_types.Range(args[0].args[0].value)
                return ast_types.Loop(range, body_dependency_map)
            
            elif isinstance(args[0].args[0], ast.Name):            
                operation = self.__map_attribute(args[0].func.id)
                dependency = args[0].args[0].id
                var = ast_types.OperationOnVariable(dependency, operation)
            
                return ast_types.Loop(var, body_dependency_map)
        
        raise TypeError("Unsupported ast node encountered in loop")
    
    def __handle_if_or_loop_body(self, node):        
        body_dependency_map = {}
        for child_node in node.body:
            self.__collect_dependencies(child_node, body_dependency_map)
        
        return dict(sorted(body_dependency_map.items(), key=lambda x: x[0][1]))
    
    def __map_node_to_custom_type(self, node):
        if isinstance(node, ast.Call):
            return self.__handle_call(node)
            
        if isinstance(node, ast.Constant):
            return ast_types.Constant(node if isinstance(node, int) else node.value)
        
        if isinstance(node, ast.Compare):
            return self.__handle_compare(node)
            
        if isinstance(node, ast.Subscript):
            return self.__handle_subscript(node)

        if isinstance(node, ast.BinOp):
            return self.__handle_binop(node)

        if isinstance(node, ast.Assign):
            return self.__handle_assign(node)
        
        if isinstance(node, ast.List):
            return self.__handle_list(node)

        if isinstance(node, ast.Name):
            return ast_types.Reference(node.id)

        if isinstance(node, ast.Attribute):
            dependency = self.__map_node_to_custom_type(node.value)
            return ast_types.Attribute(dependency, self.__map_attribute(node.attr))
                         
        if isinstance(node, ast.Return):
            if hasattr(node.value, 'id'):
                return ast_types.Assign(node.value.id)
            
            return self.__map_node_to_custom_type(node.value)

        print(ast.dump(node))
        raise TypeError("ast type not supported")
    
    def __handle_call(self, node):
        if isinstance(node.func.value, ast.Attribute):
            if isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == 'np':
                if node.func.attr == 'laplace':
                    loc = self.__map_node_to_custom_type(node.keywords[0].value)
                    scale = self.__map_node_to_custom_type(node.keywords[1].value)
                    return ast_types.Laplace(loc, scale)
         
        # TODO: Can operand both be a function and an object?            
        operand = self.__map_node_to_custom_type(node.func)
        
        # TODO: Should args also be mapped?
        return ast_types.Call(operand, node.args)
        
    def __handle_compare(self, value):
        left = self.__map_node_to_custom_type(value.left)
        # Assumes max two comparators
        left_operation = self.__map_operation(value.ops[0])
        middle_or_right = self.__map_node_to_custom_type(value.comparators[0])   
        if len(value.comparators) < 2:
            return ast_types.Compare(left, middle_or_right, left_operation)
        
        right_operation = self.__map_operation(value.ops[1])
        right = self.__map_node_to_custom_type(value.comparators[1])
            
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
    
    def __handle_binop(self, node):
        binop = ast_types.BinOp()
        binop.operation = self.__map_operation(node.op)
        binop.left = self.__map_node_to_custom_type(node.left)
        binop.right = self.__map_node_to_custom_type(node.right)
            
        return binop
    
    def __handle_assign(self, node):
        # Assumes only one target
        # IE: var1, var2 = 1, 2 not currently supported
         
        if isinstance(node.value, ast.Name):
            return ast_types.Assign(node.value.id)
         
        return self.__map_node_to_custom_type(node.value)
        
        # The below is kept for future reference
        # as it might be needed in more complex programs
        type = self.__map_node_to_custom_type(node.targets[0] if not isinstance(node.targets[0], ast.Name) else node.value)
        value = node.value.value
        return ast_types.Assign(type, value)

    def __handle_list(self, node):
        values = list(map(lambda x: self.__map_node_to_custom_type(x), node.elts))
        return ast_types.List(values)

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
    
    
    def __map_attribute(self, attribute):
        if attribute == "sum":
            return ast_types.Operation.SUM
        
        if attribute == "size" or attribute == "len":
            return ast_types.Operation.SIZE
        
        # Non 'common' attribute: Return the name of the attribute            
        return attribute
    
    def __map_operation(self, operation):
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
        
        
    