from ...custom_node import SastNode

# This models ternaries on the form 'x = 2 if y > 2 else 3'.
class IfExp(SastNode):
    condition: SastNode = None
    orelse: SastNode = None
    body: SastNode = None
    
    def __init__(self, line_number, condition, body, orelse):
        super().__init__("IfExp", line_number)
        self.condition = condition
        self.orelse = orelse
        self.body = body