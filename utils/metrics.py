import evaluate
import numpy as np

def compute_metrics(eval_pred):
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