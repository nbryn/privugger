from .functiondef_model import FunctionDef
from .. import AstTransformer
import ast


class FunctionDefTransformer(AstTransformer):
    def to_custom_model(self, node: ast.FunctionDef):
        args = list(map(lambda arg: arg.arg, node.args.args))
        body = self._collect_and_sort_by_line_number(node.body)
        return FunctionDef(node.name, node.lineno, args, body)

    def to_pymc(self, node: FunctionDef, pymc_model_builder):
        pymc_model_builder.program_functions[node.name] = (node.body, node.arguments)