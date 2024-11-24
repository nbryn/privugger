from .custom_node import SastNode
import importlib
import ast


class TransformerFactory:
    def create(self, node: ast.AST | SastNode):
        # The 'TransformerFactory' uses reflection to instantiate the correct transformer.
        # To ensure this works, the transformer must have the same name as the corresponding AST node. 
        # For example:
        # - The 'ast.Call' node must have a corresponding transformer class called `CallTransformer`.
        # - This class should be located in a folder named `call`, inside a file named `class_transformer.py`.
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

    def __get_module_path(self, node: ast.AST | SastNode):
        node_name: str = node.__class__.__name__.lower()
        if "numpy" in node_name:
            node_name = "numpy"

        if "assign" in node_name:
            node_name = "assign"

        base_path = f"privugger.white_box.transformers.{node_name}"
        if node_name in ["return", "break", "while", "for", "if"]:
            return base_path + f"_transformer.{node_name}_transformer"

        return base_path + f".{node_name}_transformer"

    def __get_transformer_name(self, node: ast.AST | SastNode):
        node_name: str = node.__class__.__name__
        if "Numpy" in node_name:
            node_name = "numpy"

        if "Assign" in node_name:
            node_name = "assign"

        return f"{node_name[0].upper() + node_name[1:]}Transformer"
