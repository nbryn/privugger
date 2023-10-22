from enum import Enum

class Operation(Enum):
    ADD = 1
    DIVIDE = 2
    MULTIPLY = 3
    EQUAL = 4
    LT = 5
    GT = 6
    SUM = 7
    SIZE = 8
    
class WrapperNode():
    value = None
    
    def __init__(self, value):
        self.value = value

class Constant():
    value = None
    
    def __init__(self, value):
        self.value = value

class Index():
    dependency_name = ""
    index = 0
    
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
        