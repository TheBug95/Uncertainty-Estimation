class BaseMethod:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    @property
    def get_model(self):
        return self.model

    @model.setter
    def set_model(self, value):
        self.model = value

    @property
    def get_data(self):
        return self.data

    @data.setter
    def set_data(self, value):
        self.data = value