class DeepEnsemble(BaseMethod):
    def __init__(self, model, data, num_samples):
        super().__init__(model, data)
        self.__num_samples = num_samples

    @property
    def num_samples(self):
        return self.__num_samples

    
    @num_samples.setter
    def num_samples(self, value):
        self.__num_samples = value

    
