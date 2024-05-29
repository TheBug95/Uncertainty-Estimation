class MCDropout(BaseMethod):
    def __init__(self, dropout_rate, num_samples, model, data):
        super().__init__(model, data)
        self.dropout_rate = dropout_rate
        self.num_samples = num_samples

    @property
    def get_dropout_rate(self):
        return self._dropout_rate


    @dropout_rate.setter
    def set_dropout_rate(self, value):
        self._dropout_rate = value


    @property
    def get_num_samples(self):
        return self._num_samples


    @num_samples.setter
    def set_num_samples(self, value):
        self._num_samples = value


    def apply_dropout(self, module, dropout_rate):
        """
        Applies dropout to a module and sets the dropout rate.

        Args:
            m (torch.nn.Module): The module to which dropout is applied.
            dropout_rate (float): The dropout rate to be set.

        Returns:
            None
        """

        if type(module) == nn.Dropout:
            # Check if the module is of type nn.Dropout
            # Set the dropout rate of the module to the specified dropout_rate
            module.p = dropout_rate

            # Set the module in training mode
            module.train()


    def make_predictions(self, model, data, droput_rate):
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

        #Define device
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        model.eval()
        model.apply(lambda module: self.apply_dropout(module, dropout_rate))

        for batch, re, ma in zip(test_dataloader, reales, mask):
            # Initialize tensor for storing predictions
            predictions_batch=torch.zeros(batch.shape[0], N)

            for i in range(N):
                print('Cycle number:', i)

                with torch.no_grad():
                    batch = batch.cuda()
                    ma = ma.cuda()
                    outputs = model(batch, ma)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                # Transpose predictions
                predictions_batch[:, i] = predictions.unsqueeze(dim=0)
                metric.add_batch(predictions=predictions, references=re)

        return predictions_batch

