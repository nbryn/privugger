from .compare_model import Compare, Compare2
from .. import AstTransformer
import ast


class CompareTransformer(AstTransformer):
    def to_custom_model(self, node: ast.Compare):
        left = self.map_to_custom_type(node.left)
        # Assumes max two comparators
        left_operation = self.map_operation(node.ops[0])
        middle_or_right = self.map_to_custom_type(node.comparators[0])
        if len(node.comparators) < 2:
            return Compare(node.lineno, left, middle_or_right, left_operation)

        right_operation = self.map_operation(node.ops[1])
        right = self.map_to_custom_type(node.comparators[1])

        return Compare2(
            node.lineno, left, left_operation, middle_or_right, right, right_operation
        )
