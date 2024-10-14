from .custom_node import CustomNode
from typing import Union
import importlib
import ast


class TransformerFactory:
    def create(self, node: Union[ast.AST, CustomNode]):
        # Use of reflection to instantiate the correct transformer.
        # For this to work the transformer must have the same name as the AST node.
        # E.g. ast.Call must have a corresponding 'CallTransformer' located in a folder named
        # 'call' in a file named 'class_transformer' with a 'CallTransformer' class.
        module_path = self.__get_module_path(node)
        try:
            transformer_module = importlib.import_module(module_path)
            transformer_class_name = self.__get_transformer_name(node)
            transformer_class = getattr(transformer_module, transformer_class_name)

            return transformer_class

        except (ModuleNotFoundError, AttributeError):
            if isinstance(node, ast.AST):
                print(ast.dump(node))

            raise RuntimeError("Error during transformer instantiation")

    def __get_module_path(self, node: Union[ast.AST, CustomNode]):
        node_name: str = node.__class__.__name__.lower()
        if "numpy" in node_name:
            node_name = "numpy"

        base_path = f"privugger.white_box.transformers.{node_name}"
        if node_name in ["return", "for", "if", "while"]:
            return base_path + f"_transformer.{node_name}_transformer"

        return base_path + f".{node_name}_transformer"

    def __get_transformer_name(self, node: Union[ast.AST, CustomNode]):
        node_name: str = node.__class__.__name__
        if "Numpy" in node_name:
            return "NumpyTransformer"

        return f"{node_name[0].upper() + node_name[1:]}Transformer"
