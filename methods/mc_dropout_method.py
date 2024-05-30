import torch
import evaluate
from methods import BaseMethod
from data import get_dataloader_by_dataset
from utils import DEVICE

class MCDropout(BaseMethod):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MCDropout, cls).__new__(cls)
        return cls._instance

    def __init__(self, dropout_rate, num_samples, model, data):
        if not hasattr(self, '_initialized'):
            super().__init__(model, data)
            self.__dropout_rate = dropout_rate
            self.__num_samples = num_samples
            self._initialized = True

    @property
    def dropout_rate(self):
        return self.__dropout_rate


    @dropout_rate.setter
    def dropout_rate(self, value):
        self.__dropout_rate = value


    @property
    def num_samples(self):
        return self.__num_samples


    @num_samples.setter
    def num_samples(self, value):
        self.__num_samples = value
        
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, new_model):
        self._model = new_model
        
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, new_data):
        self._data = new_data
        

    def __apply_dropout(self, module):
        """
        Applies dropout to a module and sets the dropout rate.

        Args:
            m (torch.nn.Module): The module to which dropout is applied.
            dropout_rate (float): The dropout rate to be set.

        Returns:
            None
        """

        if type(module) == torch.nn.Dropout:
            # Check if the module is of type nn.Dropout
            # Set the dropout rate of the module to the specified dropout_rate
            module.p = self.dropout_rate

            # Set the module in training mode
            module.train()


    def make_inference(self):
        """
        Calculates the predictions using dropout on a given model and test data.

        Args:
            model: The model for making predictions.
            test_dataloader: The dataloader for the test data.
            reales: The ground truth labels.
            mask: The mask used for dropout.
            N: The number of iterations for dropout.
            dropout_rate: The dropout rate to be applied.

        Returns:
            predictions_batch: Tensor containing the predictions for each sample and iteration.

        """
        #Define metric
        metric = evaluate.load("accuracy")
        #input_ids_batch, mask_batch, token_ids_batch,  = map(lambda t: t.to(DEVICE), data)
        labels_batch, input_ids_batch, mask_batch, _,  = get_dataloader_by_dataset(dataset = self.data)

        #Define device
        model = self.model.to(DEVICE)
        #model.to(DEVICE)

        model.eval()
        model.apply(lambda module: self.__apply_dropout(module, self.dropout_rate))

        for input_ids, labels, masks in zip(input_ids_batch, labels_batch, mask_batch):
            # Initialize tensor for storing predictions
            predictions_batch = torch.zeros(input_ids.shape[0], self.num_samples)

            for i in range(self.num_samples):
                print('Cycle number:', i)

                with torch.no_grad():
                    input_ids = input_ids.cuda()
                    masks = masks.cuda()
                    outputs = model(input_ids, masks)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                # Transpose predictions
                predictions_batch[:, i] = predictions.unsqueeze(dim=0)
                metric.add_batch(predictions = predictions, references = labels)

        return predictions_batch

    def calculate_uncertainty(self):
        pass