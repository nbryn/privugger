from .operation.operation_transformer import OperationTransformer
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
        custom_nodes = self._collect_and_sort_by_line_number(
            self.__collect_top_level_nodes(tree)
        )

        return (function_params, custom_nodes)

    def _collect_and_sort_by_line_number(self, nodes: List[ast.AST]):
        return list(
            sorted(
                map(self._map_to_custom_type, nodes),
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

    def _map_to_custom_type(self, node: ast.AST):
        if not node:
            return None

        if isinstance(node, ast.Index):
            return self._map_to_custom_type(node.value)

        # Use of reflection to instantiate the correct transformer
        # For this to work the transformer must have the same name as the AST node
        # E.g. ast.Call should have a corresponding 'CallTransformer' located in folder named 'call' with a class named 'call_transformer'
        module_path = self.__get_module_path(node)
        try:
            transformer_module = importlib.import_module(module_path)
            transformer_class_name = self.__get_transformer_name(node)
            transformer_class = getattr(transformer_module, transformer_class_name)
            
            return transformer_class().to_custom_model(node)

        except (ModuleNotFoundError, AttributeError):
            print(ast.dump(node))
            raise RuntimeError("Error during transformer instantiation")

    def __get_module_path(self, node):
        node_name = node.__class__.__name__.lower()
        base_path = f"privugger.white_box.transformers.{node_name}"

        if node_name == "return" or node_name == "if":
            return base_path + f"_transformer.{node_name}_transformer"

        return base_path + f".{node_name}_transformer"

    def __get_transformer_name(self, node):
        node_class_name = node.__class__.__name__.lower()
        if node_class_name == "binop":
            return "BinOpTransformer"

        return f"{node_class_name.capitalize()}Transformer"

    def _map_operation(self, operation):
        return OperationTransformer().to_custom_model(operation)