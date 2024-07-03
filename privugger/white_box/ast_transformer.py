import privugger.white_box.model as model
from typing import List
import ast


class AstTransformer:
    def to_custom_model(self, tree, function_def):
        function_params = list(map(lambda x: x.arg, function_def.args.args))
        custom_nodes = self.__collect_and_sort_by_line_number(
            self.__collect_top_level_nodes(tree)
        )
                                      
        return (function_params, custom_nodes)

    def __collect_and_sort_by_line_number(self, nodes: List[ast.AST]):
        return list(
            sorted(
                map(self.__map_to_custom_type, nodes),
                key=lambda node: node.line_number,
            )
        )

    def __collect_top_level_nodes(self, root: ast.AST):
        # This assumes that the root is a 'ast.FunctionDef'
        nodes = []
        for child_node in ast.iter_child_nodes(root.body[0]):
            if child_node.__class__ is not ast.arguments:
                nodes.append(child_node)

            # Assumes only one node in 'orelse'
            # TODO: Nested if's should be collected here by recursing on 'node.orelse'
            if child_node.__class__ is ast.If and len(child_node.orelse) > 0:
                nodes.append(child_node.orelse[0])

        return sorted(nodes, key=lambda node: node.lineno)

    def __map_to_custom_type(self, node: ast.AST):
        # print("NEW")
        # print(ast.dump(node))
        if not node:
            return None

        if isinstance(node, ast.If):
            # TODO: Move to 'handle_if' method
            # Note 'node.orelse' is not handled here but as a separate node
            body_custom_nodes = self.__collect_and_sort_by_line_number(node.body)
            condition = self.__map_to_custom_type(node.test)

            return model.If(node.lineno, condition, body_custom_nodes)

        if isinstance(node, ast.For):
            return self.__handle_for(node)

        if isinstance(node, ast.Call):
            return self.__handle_call(node)

        if isinstance(node, ast.Constant):
            return model.Constant(
                node.lineno, node if isinstance(node, int) else node.value
            )

        if isinstance(node, ast.Compare):
            return self.__handle_compare(node)

        if isinstance(node, ast.Index):
            # TODO: Temporary - Ensure this is handled properly
            return self.__map_to_custom_type(node.value)

        if isinstance(node, ast.Subscript):
            return self.__handle_subscript(node)

        if isinstance(node, ast.BinOp):
            return self.__handle_binop(node)

        if isinstance(node, ast.Assign):
            return self.__handle_assign(node)

        if isinstance(node, ast.List):
            return self.__handle_list(node)

        if isinstance(node, ast.Name):
            return model.Reference(node.lineno, node.id)

        if isinstance(node, ast.Attribute):
            operand = self.__map_to_custom_type(node.value)
            return model.Attribute(
                node.lineno, operand, self.__map_attribute(node.attr)
            )

        if isinstance(node, ast.Return):
            value = self.__map_to_custom_type(node.value)
            return model.Return(node.lineno, value)

        print(ast.dump(node))
        raise TypeError("ast type not supported")

    def __handle_call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            attribute = self.__map_attribute(node.func.id)
            operand = self.__map_to_custom_type(node.args[0])
            return model.Attribute(node.lineno, operand, attribute)

        if isinstance(node.func.value, ast.Attribute):
            # TODO: Handle numpy distributions in a more organized way
            if (
                isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "np"
            ):
                if node.func.attr == "laplace":
                    loc = self.__map_to_custom_type(node.keywords[0].value)
                    scale = self.__map_to_custom_type(node.keywords[1].value)
                    return model.Laplace(node.lineno, loc, scale)

        # TODO: Can operand both be a function and an object?
        operand = self.__map_to_custom_type(node.func)

        # TODO: Should args also be mapped?
        return model.Call(node.lineno, operand, node.args)

    def __handle_compare(self, node: ast.Compare):
        left = self.__map_to_custom_type(node.left)
        # Assumes max two comparators
        left_operation = self.__map_operation(node.ops[0])
        middle_or_right = self.__map_to_custom_type(node.comparators[0])
        if len(node.comparators) < 2:
            return model.Compare(node.lineno, left, middle_or_right, left_operation)

        right_operation = self.__map_operation(node.ops[1])
        right = self.__map_to_custom_type(node.comparators[1])

        return model.Compare2(
            node.lineno, left, left_operation, middle_or_right, right, right_operation
        )

    def __handle_subscript(self, node: ast.Subscript):
        if isinstance(node.slice, ast.Index):
            index = (
                node.slice.value.value
                if hasattr(node.slice.value, "value")
                else node.slice.value.id
            )
            return model.Index(node.lineno, node.value.id, index)

        lower = self.__map_to_custom_type(node.slice.lower)
        upper = self.__map_to_custom_type(node.slice.upper)
        dependency_name = node.value.id

        return model.Subscript(node.lineno, dependency_name, lower, upper)

    def __handle_binop(self, node: ast.BinOp):
        operation = self.__map_operation(node.op)
        left = self.__map_to_custom_type(node.left)
        right = self.__map_to_custom_type(node.right)

        return model.BinOp(node.lineno, left, right, operation)

    def __handle_assign(self, node: ast.Assign):
        # Assumes only one target
        # IE: var1, var2 = 1, 2 not currently supported
        temp_node = node.targets[0]
        while not isinstance(temp_node, ast.Name):
            temp_node = temp_node.value

        if isinstance(node.targets[0], ast.Subscript):
            index = self.__map_to_custom_type(node.targets[0].slice)
            value = self.__map_to_custom_type(node.value)

            return model.AssignIndex(temp_node.id, node.lineno, value, index)

        value = self.__map_to_custom_type(node.value)
        return model.Assign(temp_node.id, node.lineno, value)

    def __handle_list(self, node: ast.List):
        values = list(map(self.__map_to_custom_type, node.elts))
        return model.ListNode(node.lineno, values)

    def __handle_for(self, node: ast.For):
        # Loops are not well supported in PyMC, meaning they shouldn't be translated
        # into PyMC variables but instead function as a 'normal' python loop.
        # Only standard 'for i in range()' loops supported for now
        body = self.__collect_and_sort_by_line_number(node.body)

        # We only have stop like 'range(stop)'
        if len(node.iter.args) == 1:
            start = model.Constant(node.lineno, 0)
            stop = self.__map_to_custom_type(node.iter.args[0])
            return model.Loop(node.lineno, start, stop, body)

        # We have both start and stop like 'range(start, stop)'
        start = self.__map_to_custom_type(node.iter.args[0])
        stop = self.__map_to_custom_type(node.iter.args[1])

        return model.Loop(node.lineno, start, stop, body)

    def __map_attribute(self, attribute_name):
        if attribute_name == "sum":
            return model.Operation.SUM

        if attribute_name == "size" or attribute_name == "len":
            return model.Operation.SIZE

        # Non 'common' attribute: Return the name of the attribute
        return attribute_name

    def __map_operation(self, operation):
        if operation == "sum":
            return model.Operation.SUM

        if operation == "size" or operation == "len":
            return model.Operation.SIZE

        if isinstance(operation, ast.Add):
            return model.Operation.ADD

        if isinstance(operation, ast.Div):
            return model.Operation.DIVIDE

        if isinstance(operation, ast.Mult):
            return model.Operation.MULTIPLY

        if isinstance(operation, ast.Lt):
            return model.Operation.LT

        if isinstance(operation, ast.LtE):
            return model.Operation.LTE

        if isinstance(operation, ast.Gt):
            return model.Operation.GT

        if isinstance(operation, ast.GtE):
            return model.Operation.GTE

        print(operation)
        raise TypeError("Unknown operation")
