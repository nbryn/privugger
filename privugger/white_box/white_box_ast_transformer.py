from .transformers.operation.operation_transformer import OperationTransformer
from .transformer_factory import TransformerFactory
from .custom_node import CustomNode
from .method import Method
from typing import List
import ast


class WhiteBoxAstTransformer:
    transformer_factory = TransformerFactory()
    program_variables = {}
    program_functions = {}
    global_priors = []
    num_elements = 0
    pymc_model = None
    method = None

    def __init__(self, global_priors=[], num_elements=0, pymc_model=None, method=Method.PYMC):
        self.global_priors = global_priors
        self.num_elements = num_elements
        self.pymc_model = pymc_model
        self.method = method

    def transform(self, abstract_syntax_tree, function_def):
        # Construct custom model
        program_arguments = list(map(lambda x: x.arg, function_def.args.args))
        custom_nodes = self.collect_and_sort_by_line_number(
            self.__collect_top_level_nodes(abstract_syntax_tree)
        )

        # Map top level function args to PyMC. Args must be of type 'pv.Distribution'
        for index, arg_name in enumerate(program_arguments):
            # TODO: Can we have more than one argument?
            self.program_variables[arg_name] = (
                self.global_priors[index],
                self.num_elements,
            )

        # Construct white-box model using selected method
        for node in custom_nodes:
            if self.method == Method.PYMC:
                self.to_pymc(node)


    def collect_and_sort_by_line_number(self, nodes: List[ast.AST]):
        return list(
            sorted(
                map(self.to_custom_model, nodes),
                key=lambda node: node.line_number,
            )
        )

    def __collect_top_level_nodes(self, root: ast.AST):
        # Assumes that the root is a 'ast.FunctionDef'
        nodes = []
        for child_node in ast.iter_child_nodes(root.body[0]):
            if child_node.__class__ is not ast.arguments:
                nodes.append(child_node)

            # Assumes only one node in 'orelse'
            # TODO: Nested if's should be collected here by recursing on 'node.orelse'
            if child_node.__class__ is ast.If and len(child_node.orelse) > 0:
                nodes.append(child_node.orelse[0])

        return sorted(nodes, key=lambda node: node.lineno)

    def to_custom_model(self, node: ast.AST):
        if not node:
            return None

        transformer_class = self.transformer_factory.create(node)
        return transformer_class().to_custom_model(node)

    def to_pymc(self, node: CustomNode, condition=None, in_function=False):
        # print("MAP TO")
        # print(type(node))
        # print(type(self.transformer_factory.create(node)))

        transformer_class = self.transformer_factory.create(node)
        return transformer_class(self.program_variables, self.program_functions, self.pymc_model).to_pymc(
            node, condition, in_function
        )

    def _map_operation(self, operation):
        return OperationTransformer().to_custom_model(operation)

    def _to_pymc_operation(self, operation, operand, right=None):
        return OperationTransformer().to_pymc(operation, operand, right)