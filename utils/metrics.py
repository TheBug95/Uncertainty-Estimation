import evaluate
import numpy as np
from scipy.stats import entropy

def compute_accuracy_metrics(eval_pred):
    """
    Compute accuracy metric for evaluation.

    Parameters:
    eval_pred (tuple): A tuple containing logits and labels.

    Returns:
    dict: A dictionary containing the accuracy of the predictions.
    """
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def calculate_entropy(predictions):
    """
    Calculate the entropy for a set of predictions.

    Args:
        predictions (numpy.ndarray): A 2D array of shape (n_samples, n_classes) containing the predicted probabilities for each class.

    Returns:
        tuple:
            - entropia (numpy.ndarray): A 1D array containing the entropy for each sample.
            - pred (numpy.ndarray): A 1D array containing the binary predictions for each sample.
    
    The function performs the following steps:
        1. Calculate the probability of class 1 for each sample.
        2. Concatenate the probabilities of class 1 and class 0.
        3. Calculate the entropy for each sample using the concatenated probabilities.
        4. Recode the probabilities of class 1 into binary predictions.
    """

    # Calculate the probability of class 1 for each sample
    prob1 = np.sum(predictions, axis=-1) / predictions.shape[1]

    # Concatenate the probabilities of class 1 and class 0
    arrProbs = np.hstack((prob1.reshape(-1, 1), (1 - prob1).reshape(-1, 1)))

    # Calculate the entropy for each sample using scipy.stats.entropy
    entropia = entropy(arrProbs, base=2, axis=1)

    # Recode the probabilities of class 1 into binary predictions (0 or 1)
    pred = np.where(prob1 > 0.5, 1, 0)

    return entropia, pred


    