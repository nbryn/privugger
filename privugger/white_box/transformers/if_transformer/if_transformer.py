from ...ast_transformer import AstTransformer
from ..break_transformer.break_model import Break
from ...custom_node import CustomNode
from .if_model import If
from typing import List
import ast


class IfTransformer(AstTransformer):
    def to_custom_model(self, node: ast.If):
        orelse = super().collect_and_sort_by_line_number(node.orelse)
        body = super().collect_and_sort_by_line_number(node.body)
        condition = super().to_custom_model(node.test)
        has_break_in_body = any(isinstance(child_node, Break) for child_node in body)
        
        # We can have multiple if's here
        child_ifs = list(filter(lambda child_node: isinstance(child_node, If), body))
        if_node = If(node.lineno, condition, body, orelse, has_break_in_body)
        for child_if in child_ifs:
            child_if.parent_if = if_node
        
        return if_node

    # AssignTransformer handles conditionally assigning values
    # depending on whether the condition is true or not.
    def to_pymc(self, node: If, conditions: dict, in_function):
        condition = super().to_pymc(node.condition, conditions, in_function)
        updated_conditions = conditions.copy()
        updated_conditions[node.name_with_line_number] = condition
        for child_node in node.body:
            if not isinstance(child_node, Break):
                super().to_pymc(child_node, updated_conditions, in_function)

        updated_conditions[node.name_with_line_number] = ~condition
        for child_node in node.orelse:
            if not isinstance(child_node, Break):
                super().to_pymc(child_node, updated_conditions, in_function)

        