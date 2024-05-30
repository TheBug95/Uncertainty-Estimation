from abc import ABC, abstractmethod 

class BaseMethod(ABC):
    @abstractmethod
    def __init__(self, model, data):
        self._model = model
        self._data = data

    @property
    @abstractmethod
    def model(self):
        pass

    @model.setter
    @abstractmethod
    def model(self, value):
        pass

    @property
    @abstractmethod
    def data(self):
        pass

    @data.setter
    @abstractmethod
    def data(self, value):
        pass
        
    @abstractmethod
    def calculate_uncertainty(self):
        pass
    
    @abstractmethod
    def make_inference(self):
        pass
    
    
    