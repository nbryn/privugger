from abc import abstractmethod
class CustomNode:
    name_with_line_number = ""
    line_number = -1
    name = ""

    def __init__(self, name, line_number):
        self.name_with_line_number = f"{name} - {str(line_number)}"
        self.line_number = line_number
        self.name = name
    
    @abstractmethod
    def to_pymc(self):
        pass