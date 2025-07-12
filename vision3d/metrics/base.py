from abc import ABC, abstractmethod

class Metric(ABC):
    """
    Abstract base class for defining a metric.

    Methods:
        reset():
            Reset the metric to its initial state. 
        update(preds, targets):
            Update the metric with predictions and targets. 

        compute() -> dict:
            Compute and return the final metric values as a dictionary. 
    """
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, preds, targets):
        pass

    @abstractmethod
    def compute(self) -> dict:
        pass

