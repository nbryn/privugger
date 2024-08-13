import privugger.white_box.model as model

class BinOp(model.CustomNode):
    operation: model.Operation = None
    right: model.CustomNode = None
    left: model.CustomNode = None

    def __init__(self, line_number, left, right, operation):
        model.CustomNode.__init__(self, "BinOp", line_number)
        self.operation = operation
        self.right = right
        self.left = left