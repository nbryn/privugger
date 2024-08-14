#from privugger.white_box.transformers import *
from abc import abstractmethod
from typing import List
import importlib
import ast


class AstTransformer:
    @abstractmethod
    def to_custom_model(self):
        pass

    def transform(self, tree, function_def):
        function_params = list(map(lambda x: x.arg, function_def.args.args))
        custom_nodes = self.collect_and_sort_by_line_number(
            self.__collect_top_level_nodes(tree)
        )

        return (function_params, custom_nodes)

    def collect_and_sort_by_line_number(self, nodes: List[ast.AST]):
        return list(
            sorted(
                map(self.map_to_custom_type, nodes),
                key=lambda node: node.line_number,
            )
        )

    def __collect_top_level_nodes(self, root: ast.AST):
        # Assumes that the root is a 'ast.FunctionDef'
        nodes = []
        for child_node in ast.iter_child_nodes(root.body[0]):
            if child_node.__class__ is not ast.arguments:
                nodes.append(child_node)

            # Assumes only one node in 'orelse'
            # TODO: Nested if's should be collected here by recursing on 'node.orelse'
            if child_node.__class__ is ast.If and len(child_node.orelse) > 0:
                nodes.append(child_node.orelse[0])

        return sorted(nodes, key=lambda node: node.lineno)

    def map_to_custom_type(self, node: ast.AST):
        if not node:
            return None
        
        # Use reflection to instantiate the correct transformer
        # For this to work the transformer must have the same name ast the AST node
        # E.g. ast.Call must have a corresponding CallTransformer
        # Get the class name of the AST node (e.g., If, For, Call)
        node_class_name = node.__class__.__name__

        # Construct the transformer class name (e.g., IfTransformer, ForTransformer)
        transformer_class_name = f"{node_class_name}Transformer"

        # Construct the module path based on the node class name (e.g., if.if_transformer)
        module_path = f"{node_class_name.lower()}.{node_class_name.lower()}_transformer"
        print(module_path)

        try:
        # Import the module dynamically from the corresponding folder
            transformer_module = importlib.import_module(module_path)
            print("GOT HERE")
            
            # Get the transformer class from the imported module
            transformer_class = getattr(transformer_module, transformer_class_name)
            
            transformer = transformer_class()
            return transformer.to_custom_model(node)

        except (ModuleNotFoundError, AttributeError):
            if isinstance(node, ast.Index):
            # Handle special case for ast.Index
                return self.map_to_custom_type(node.value)
        
        print(ast.dump(node))
        raise TypeError(f"AST type {node_class_name} not supported")

        # print("NEW")
        # print(ast.dump(node))

        if isinstance(node, ast.If):
            return IfTransformer().to_custom_model(node)

        if isinstance(node, ast.For):
            return ForTransformer().to_custom_model(node)

        if isinstance(node, ast.Call):
            return CallTransformer().to_custom_model(node)

        if isinstance(node, ast.Constant):
            return ConstantTransformer().to_custom_model(node)

        if isinstance(node, ast.Compare):
            return CompareTransformer().to_custom_model(node)

        if isinstance(node, ast.Index):
            # TODO: Temporary - Ensure this is handled properly
            return self.map_to_custom_type(node.value)

        if isinstance(node, ast.Subscript):
            return SubscriptTransformer().to_custom_model(node)

        if isinstance(node, ast.BinOp):
            return BinOpTransformer().to_custom_model(node)

        if isinstance(node, ast.UnaryOp):
            return UnaryOpTransformer().to_custom_model(node)

        if isinstance(node, ast.Assign):
            return AssignTransformer().to_custom_model(node)

        if isinstance(node, ast.List):
            return ListTransformer().to_custom_model(node)

        if isinstance(node, ast.Name):
            return NameTransformer().to_custom_model(node)

        if isinstance(node, ast.Attribute):
            return AttributeTransformer().to_custom_model(node)

        if isinstance(node, ast.FunctionDef):
            return FunctionTransformer().to_custom_model(node)

        if isinstance(node, ast.Return):
            return ReturnTransformer().to_custom_model(node)

        print(ast.dump(node))
        raise TypeError("ast type not supported")

    def map_operation(self, operation):
        return OperationTransformer().to_custom_model(operation)
