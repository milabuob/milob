from abc import ABC, abstractmethod


class BaseAnalysis(ABC):
    def __init__(self, dataset):
        self.dataset = dataset
        self.results = None
    
    
    @abstractmethod
    def fit(self, **kwargs):
        """Fit this analysis to its dataset."""
        pass
    
    
    def __repr__(self):
        return f"<{self.__class__.__name__} Analysis>"