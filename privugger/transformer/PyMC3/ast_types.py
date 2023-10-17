class Variable():
    dependency_name = ""
    function = ""
    
    def __init__(self, dependency_name, function):
        self.dependency_name = dependency_name
        self.function = function


class Subscript():
    dependency_name = ""
    lower = None
    upper = None
    
    
class BinOp():
    operation = None
    left: Variable = None
    right: Variable = None
    

class Call():
    operation = None
    left_side: Variable = None
    right_side: Variable = None
    
    def __init__(self, operation, left_side, right_side):
        self.operation = operation
        self.left_side = left_side
        self.right = right_side