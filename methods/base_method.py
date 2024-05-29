from abc import ABC, abstractmethod 

class BaseMethod(ABC):
    @abstractmethod
    def __init__(self, model, data):
        self.__model = model
        self.__data = data

    @property
    @abstractmethod
    def model(self):
        return self.__model

    @model.setter
    @abstractmethod
    def model(self, value):
        self.__model = value

    @property
    @abstractmethod
    def data(self):
        return self.__data

    @data.setter
    @abstractmethod
    def data(self, value):
        self.__data = value
        
    @abstractmethod
    def calculate_uncertainty(self):
        pass
    
    @abstractmethod
    def make_inference(self):
        pass
    
    
    