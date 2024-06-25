import privugger.transformer.PyMC3.ast_types as ast_types
from typing import List
import ast

class AST_Transformer():
    def to_custom_model(self, tree, function_def):
        function_params = list(map(lambda x: x.arg, function_def.args.args))
        custom_nodes = self.__collect_and_sort_by_line_number(self.__collect_top_level_nodes(tree))
        
        return (function_params, custom_nodes)
    
    def __collect_and_sort_by_line_number(self, nodes: List[ast.AST]):
        return list(
            sorted(
                map(self.__map_node_to_custom_type, nodes),
                key=lambda node: node.line_number
                )
            )
    
    def __collect_top_level_nodes(self, root: ast.AST):
        # This assumes that the root is a 'ast.FunctionDef'
        nodes = []
        for child_node in ast.iter_child_nodes(root.body[0]):
            if child_node.__class__ is not ast.arguments:
                nodes.append(child_node)
            
            if child_node.__class__ is ast.If and len(child_node.orelse) > 0:
                # Assumes only one node in 'orelse'
                nodes.append(child_node.orelse[0])
            
        return sorted(nodes, key=lambda node: node.lineno)
    
    def __map_node_to_custom_type(self, node: ast.AST):
        if not node: return None
        
        if isinstance(node, ast.If):            
            # TODO: Move to 'handle_if' method
            # Note 'node.orelse' is not handled here but as a separate node
            body_custom_nodes = self.__collect_and_sort_by_line_number(node.body)            
            condition = self.__map_node_to_custom_type(node.test)

            return ast_types.If(node.lineno, condition, body_custom_nodes)   
            
        if isinstance(node, ast.For):
            # Loops are not well supported in PyMC.
            # So loops shouldn't be translated into PyMC variables
            # but instead function as a 'normal' python loop.
            # Only standard 'for i in range()' loops supported for now
            body_dependency_map = self.__collect_and_sort_by_line_number(node)
            args = node.iter.args
            if isinstance(args[0], ast.Call):
                if isinstance(args[0].args[0], ast.Constant):
                    range = ast_types.Range(node.lineno, args[0].args[0].value)
                    return ast_types.Loop(node.lineno, range, body_dependency_map)
                
                elif isinstance(args[0].args[0], ast.Name):            
                    attribute = self.__map_attribute(args[0].func.id)
                    dependency = args[0].args[0].id
                    var = ast_types.Attribute(node.lineno, dependency, attribute)
                
                    return ast_types.Loop(node.lineno, var, body_dependency_map)
            
            raise TypeError("Unsupported ast node encountered in loop")            
        
        if isinstance(node, ast.Call):
            return self.__handle_call(node)
            
        if isinstance(node, ast.Constant):
            return ast_types.Constant(node.lineno, node if isinstance(node, int) else node.value)
        
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
            return ast_types.Reference(node.lineno, node.id)

        if isinstance(node, ast.Attribute):
            dependency = self.__map_node_to_custom_type(node.value)
            return ast_types.Attribute(node.lineno, dependency, self.__map_attribute(node.attr))
                         
        if isinstance(node, ast.Return):
            value = self.__map_node_to_custom_type(node.value)
            return ast_types.Return(node.lineno, value)

        print(ast.dump(node))
        raise TypeError("ast type not supported")
    
    def __handle_call(self, node: ast.Call):
        if isinstance(node.func.value, ast.Attribute):
            if isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == 'np':
                if node.func.attr == 'laplace':
                    loc = self.__map_node_to_custom_type(node.keywords[0].value)
                    scale = self.__map_node_to_custom_type(node.keywords[1].value)
                    return ast_types.Laplace(node.lineno, loc, scale)
         
        # TODO: Can operand both be a function and an object?            
        operand = self.__map_node_to_custom_type(node.func)
        
        # TODO: Should args also be mapped?
        return ast_types.Call(node.lineno, operand, node.args)
        
    def __handle_compare(self, node: ast.Compare):
        left = self.__map_node_to_custom_type(node.left)
        # Assumes max two comparators
        left_operation = self.__map_operation(node.ops[0])
        middle_or_right = self.__map_node_to_custom_type(node.comparators[0])   
        if len(node.comparators) < 2:
            return ast_types.Compare(node.lineno, left, middle_or_right, left_operation)
        
        right_operation = self.__map_operation(node.ops[1])
        right = self.__map_node_to_custom_type(node.comparators[1])
            
        return ast_types.Compare2(node.lineno, left, left_operation, middle_or_right, right, right_operation)
    
    def __handle_subscript(self, node: ast.Subscript):
        if isinstance(node.slice, ast.Index):
            return ast_types.Index(node.lineno, node.value.id, node.slice.value.value if hasattr(node.slice.value, 'value') else node.slice.value.id)
        
        lower = self.__map_node_to_custom_type(node.slice.lower)
        upper = self.__map_node_to_custom_type(node.slice.upper)
        dependency_name = node.value.id
        
        return ast_types.Subscript(node.lineno, dependency_name, lower, upper)
    
    def __handle_binop(self, node: ast.BinOp):
        operation = self.__map_operation(node.op)
        left = self.__map_node_to_custom_type(node.left)
        right = self.__map_node_to_custom_type(node.right)
            
        return ast_types.BinOp(node.lineno, left, right, operation)
    
    def __handle_assign(self, node: ast.Assign):
        # Assumes only one target
        # IE: var1, var2 = 1, 2 not currently supported
        if isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id 
            value = self.__map_node_to_custom_type(node.value)
            return ast_types.Assign(var_name, node.lineno, value)
        
        raise TypeError('Assign: Unsupported target')
        # The below is kept for future reference
        # as it might be needed in more complex programs
        type = self.__map_node_to_custom_type(node.targets[0] if not isinstance(node.targets[0], ast.Name) else node.value)
        value = node.value.value
        return ast_types.Assign(type, value)

    def __handle_list(self, node: ast.List):
        values = list(map(lambda x: self.__map_node_to_custom_type(x), node.elts))
        return ast_types.ListNode(node.lineno, values)
    
    def __map_attribute(self, attribute_name):
        if attribute_name == "sum":
            return ast_types.Operation.SUM
        
        if attribute_name == "size" or attribute_name == "len":
            return ast_types.Operation.SIZE
        
        # Non 'common' attribute: Return the name of the attribute            
        return attribute_name
    
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
        
        
    