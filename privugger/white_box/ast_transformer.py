from .transformer_factory import TransformerFactory
from .custom_node import CustomNode
from .method import Method
from typing import List
import ast


class AstTransformer:
    transformer_factory = TransformerFactory()
    program_variables = {}
    program_functions = {}
    global_priors = []
    num_elements = 0
    pymc_model = None
    method = None

    def __init__(
        self, global_priors=[], num_elements=0, pymc_model=None, method=Method.PYMC
    ):
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

            else:
                raise RuntimeError("Unsupported method")

    def collect_and_sort_by_line_number(self, nodes: List[ast.AST]):
        return list(
            sorted(
                map(lambda node: AstTransformer.to_custom_model(self, node), nodes),
                key=lambda node: node.line_number,
            )
        )

    def __collect_top_level_nodes(self, root: ast.AST):
        # Assumes that root is 'ast.FunctionDef'
        nodes = []
        for child_node in ast.iter_child_nodes(root.body[0]):
            if child_node.__class__ is not ast.arguments:
                nodes.append(child_node)

        return sorted(nodes, key=lambda node: node.lineno)

    def to_custom_model(self, node: ast.AST):
        if not node:
            return None

        transformer_class = self.transformer_factory.create(node)
        return transformer_class().to_custom_model(node)

    def to_pymc(self, node: CustomNode, condition=None, in_function=False):
        transformer_class = self.transformer_factory.create(node)
        return transformer_class(
            self.program_variables, self.program_functions, self.pymc_model
        ).to_pymc(node, condition, in_function)
