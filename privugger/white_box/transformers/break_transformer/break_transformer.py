from ...ast_transformer import AstTransformer
from .break_model import Break
import ast


class BreakTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Break):
        return Break(node.lineno)

    def to_pymc(self, node: Break, __, ___):
        raise TypeError(
            "Should not get here: Break is handled in the 'While' and 'For' transformers"
        )
