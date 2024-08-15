
import importlib
import ast


class TransformerFactory:
    def create_transformer(self, node):
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
            if isinstance(node, ast.AST):
                print(ast.dump(node))
            
            raise RuntimeError("Error during transformer instantiation")

    def __get_module_path(self, node):
        node_name: str = node.__class__.__name__.lower()
        if "numpy" in node_name:
            node_name = "numpy"
        
        base_path = f"privugger.white_box.transformers.{node_name}"
        if node_name == "return" or node_name == "if" or node_name == "for":
            return base_path + f"_transformer.{node_name}_transformer"

        return base_path + f".{node_name}_transformer"

    def __get_transformer_name(self, node):
        node_name = node.__class__.__name__.lower()
        if "numpy" in node_name:
            return "NumpyTransformer"
        
        if node_name == "binop":
            return "BinOpTransformer"

        return f"{node_name.capitalize()}Transformer"
