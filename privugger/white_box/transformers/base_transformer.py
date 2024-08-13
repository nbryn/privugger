from abc import ABC, abstractmethod
from .. import AstTransformer

class Transformer(ABC):
    ast_transformer: AstTransformer
    
    def __init__(self, ast_transformer):
        self.ast_transformer = ast_transformer
    
    @abstractmethod
    def Transform(self):
        pass