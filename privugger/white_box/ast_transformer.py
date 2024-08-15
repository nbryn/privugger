from .transformers.operation.operation_transformer import OperationTransformer
from .transformer_factory import TransformerFactory
from abc import abstractmethod
from typing import List
import ast


class AstTransformer:
    transformer_factory = TransformerFactory()
    
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

        transformer = self.transformer_factory.create(node)
        return transformer.to_custom_model(node)

    def _map_operation(self, operation):
        return OperationTransformer().to_custom_model(operation)