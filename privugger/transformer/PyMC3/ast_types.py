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

class Laplace():
    loc = None
    scale = None
    
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        
class Reference():
    dependency_name = ""
    
    def __init__(self, dependency_name):
        self.dependency_name = dependency_name

class Constant():
    value = None
    
    def __init__(self, value):
        self.value = value

class Assign():
    type = None
    value = None

    def __init__(self, type, value=None):
        self.type = type
        self.value = value
        
class List():
    values = []

    def __init__(self, values):
        self.values = values
        
class Index():
    dependency_name = ""
    index = None
    
    def __init__(self, dependency_name, index):
        self.dependency_name = dependency_name
        self.index = index
        
class If():
    test = None
    body = None
    else_branch = None
    
    def __init__(self, test, body, else_branch=None):
        self.test = test
        self.body = body
        else_branch = else_branch
    
class Compare():
    left = None
    right = None
    operation: Operation = None
    
    def __init__(self, left, right, operation):
        self.left = left
        self.right = right
        self.operation = operation

class Compare2():
    left = None
    left_operation: Operation = None
    middle = None
    right = None
    right_operation: Operation = None
    
    def __init__(self, left, left_operation, middle, right, right_operation):
        self.left = left
        self.left_operation = left_operation
        self.middle = middle
        self.right = right
        self.right_operation = right_operation

class Subscript():
    dependency_name = ""
    lower = None
    upper = None
       
class BinOp():
    operation: Operation = None
    left = None
    right = None

# Represents 'object.attribute'
# Attribute can be a 'normal' operation (like sum) or a custom attribute
# If its a operation it will be of type operation otherwise it will be a string
class Attribute():
    operand: Reference = None
    attribute = None
    
    def __init__(self, operand, attribute):
        self.operand = operand
        self.attribute = attribute
    
class Call():
    operand = None
    arguments = []
    
    def __init__(self, operand, arguments):
        self.operand = operand
        self.arguments = arguments

class Loop():
    condition = None
    body = None
    
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class Range():
    start = 0
    stop = None
    
    def __init__(self, stop, start = 0):
        self.start = start
        self.stop = stop
    
    
        