from ...white_box_ast_transformer  import WhiteBoxAstTransformer
from .functiondef_model import FunctionDef
import ast


class FunctionDefTransformer(WhiteBoxAstTransformer):
    def to_custom_model(self, node: ast.FunctionDef):
        args = list(map(lambda arg: arg.arg, node.args.args))
        body = self.collect_and_sort_by_line_number(node.body)
        return FunctionDef(node.name, node.lineno, args, body)

    def to_pymc(self, node: FunctionDef, _, __):
        self.program_functions[node.name] = (node.body, node.arguments)