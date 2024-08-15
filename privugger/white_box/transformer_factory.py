from .ast_transformer import AstTransformer
import importlib
import ast


class TransformerFactory:
    def create(self, node: ast.AST) -> AstTransformer:
        # Use of reflection to instantiate the correct transformer
        # For this to work the transformer must have the same name as the AST node
        # E.g. ast.Call should have a corresponding 'CallTransformer' located in folder named 'call' with a class named 'call_transformer'
        module_path = self.__get_module_path(node)
        try:
            transformer_module = importlib.import_module(module_path)
            transformer_class_name = self.__get_transformer_name(node)
            transformer_class = getattr(transformer_module, transformer_class_name)

            return transformer_class()

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
