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
    
class WrapperNode():
    value = None
    
    def __init__(self, value):
        self.value = value

class Constant():
    value = None
    
    def __init__(self, value):
        self.value = value

class Assign():
    temp = None
    value = None

    def __init__(self, temp, value=None):
        self.temp = temp
        self.value = value
        
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

# Rename
class Variable():
    dependency_name = ""
    operation: Operation = None
    
    def __init__(self, dependency_name, operation):
        self.dependency_name = dependency_name
        self.operation = operation

class Subscript():
    dependency_name = ""
    lower = None
    upper = None
       
class BinOp():
    operation: Operation = None
    left: Variable = None
    right: Variable = None
    
class Call():
    operand = None
    operation: Operation = None
    
    def __init__(self, operand, operation):
        self.operand = operand
        self.operation = operation

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
    
    
        