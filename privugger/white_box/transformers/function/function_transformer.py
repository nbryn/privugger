from .function_model import FunctionDef
from .. import AstTransformer
import ast


class FunctionTransformer(AstTransformer):
    def to_custom_model(self, node: ast.FunctionDef):
        args = list(map(lambda arg: arg.arg, node.args.args))
        body = self._collect_and_sort_by_line_number(node.body)
        return FunctionDef(node.name, node.lineno, args, body)
