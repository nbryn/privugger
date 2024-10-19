from ...ast_transformer import AstTransformer
from ..while_transformer.while_model import While
from ..for_transformer.for_model import For
from ..if_transformer.if_model import If
from ...custom_node import CustomNode
from .break_model import Break
import ast


class BreakTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Break):
        return Break(node.lineno)

    def to_pymc(self, node: Break, __, ___):
        raise TypeError(
            "Should not get here: Break is handled in the 'While' and 'For' transformers"
        )

    def should_break(self, node: CustomNode) -> bool:
        if isinstance(node, Break):
            return True

        if not isinstance(node, If | For | While):
            return False

        ifs_with_break_in_body = []
        self.__collect_if_nodes_with_break_in_body(node, ifs_with_break_in_body)
        for if_node in ifs_with_break_in_body:
            current = if_node
            conditions = []
            while current is not None:
                conditions.append(super().to_pymc(current.condition))
                current = current.parent_if

            if all(condition.eval() for condition in conditions):
                return True

        return False

    def __collect_if_nodes_with_break_in_body(
        self, node: If | For | While, nodes: list[If]
    ):
        if node.has_break_in_body:
            nodes.append(node)

        for child_node in node.body:
            if isinstance(child_node, If | For | While):
                self.__collect_if_nodes_with_break_in_body(child_node, nodes)
