from typing import List
from enum import Enum

class Operation(Enum):
    ADD = 1
    DIVIDE = 2
    MULTIPLY = 3
    EQUAL = 4
    LT = 5
    LTE = 6
    GT = 7
    GTE = 8
    SUM = 9
    SIZE = 10

class CustomNode():
    name_with_line_number = ''
    line_number = -1
    name = ''
    
    def __init__(self, name, line_number):
        self.name_with_line_number = f'{name} - {str(line_number)}'
        self.line_number = line_number
        self.name = name
        
class Assign(CustomNode):
    value = None

    def __init__(self, variable_name, line_number, value):
        CustomNode.__init__(self, variable_name, line_number)
        self.value = value

class Return(CustomNode):
    value = None
    
    def __init__(self, line_number, value):
        CustomNode.__init__(self, 'return', line_number)
        self.value = value
        
class If(CustomNode):
    body: List[CustomNode] = []
    condition = None
    
    def __init__(self, line_number, condition, body):
        CustomNode.__init__(self, 'if', line_number)
        self.condition = condition
        self.body = body
    
class Loop(CustomNode):
    body: List[CustomNode] = []
    condition = None
    
    def __init__(self, line_number, condition, body):
        CustomNode.__init__('Loop', line_number)
        self.condition = condition
        self.body = body

class Distribution(CustomNode):
    scale = None
    loc = None
    
    def __init__(self, name, line_number, scale, loc):
        CustomNode.__init__(self, name, line_number)
        self.scale = scale
        self.loc = loc

class Laplace(Distribution):
    def __init__(self, line_number, loc, scale):
        Distribution.__init__(self, 'Laplace', line_number, scale, loc)
        
class Reference(CustomNode):
    reference_to = ''
    
    def __init__(self, line_number, reference_to):
        CustomNode.__init__(self, 'Reference', line_number)
        self.reference_to = reference_to

class Constant(CustomNode):
    value = None
    
    def __init__(self, line_number, value):
        CustomNode.__init__(self, 'Constant', line_number)
        self.value = value

class ListNode(CustomNode):
    values = []

    def __init__(self, line_number, values):
        CustomNode.__init__(self, 'List', line_number)
        self.values = values
        
class Index(CustomNode):
    dependency_name = ''
    index = None
    
    def __init__(self, line_number, dependency_name, index):
        CustomNode.__init__(self, 'Index', line_number)
        self.dependency_name = dependency_name
        self.index = index
    
# TODO: Compare doesn't have to be a operation, can also be function call    
class Compare(CustomNode):
    operation: Operation = None
    right = None
    left = None
    
    def __init__(self, line_number, left, right, operation):
        CustomNode.__init__(self, 'Compare', line_number)
        self.left = left
        self.right = right
        self.operation = operation

# TODO: Compare2 doesn't have to be a operation, can also be function call    
class Compare2(CustomNode):
    right_operation: Operation = None
    left_operation: Operation = None
    middle = None
    right = None
    left = None
    
    def __init__(self, line_number, left, left_operation, middle, right, right_operation):
        CustomNode.__init__(self, 'Compare2', line_number)
        self.right_operation = right_operation
        self.left_operation = left_operation
        self.middle = middle
        self.right = right
        self.left = left

class Subscript(CustomNode):
    dependency_name = ''
    lower = None
    upper = None
    
    def __init__(self, line_number, dependency_name, lower, upper):
        CustomNode.__init__(self, 'Subscript', line_number)
        self.dependency_name = dependency_name
        self.lower = lower
        self.upper = upper
    
       
class BinOp(CustomNode):
    operation: Operation = None
    right = None
    left = None
    
    def __init__(self, line_number, left, right, operation):
        CustomNode.__init__(self, 'BinOp', line_number)
        self.operation = operation
        self.right = right
        self.left = left
    

# Represents 'object.attribute'
# Attribute can be a 'normal' operation (like sum) or a custom attribute
# If its a operation it will be of type operation otherwise it will be a string
class Attribute(CustomNode):
    operand: Reference = None
    attribute = None
    
    def __init__(self, line_number, operand, attribute):
        CustomNode.__init__(self, 'Attribute', line_number)
        self.attribute = attribute
        self.operand = operand
        
    
class Call(CustomNode):
    arguments = []
    operand = None
    
    def __init__(self, line_number, operand, arguments):
        CustomNode.__init__(self, 'Call', line_number)
        self.arguments = arguments
        self.operand = operand
        

class Range(CustomNode):
    stop = None
    start = 0
    
    def __init__(self, line_number, stop, start = 0):
        CustomNode.__init__(self, 'Range', line_number)
        self.start = start
        self.stop = stop
    
    
        