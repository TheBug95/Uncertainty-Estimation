import torch

from utils import (
    compute_accuracy_metrics, 
    PATH_MODEL_HF, 
    PATH_SAVE_MODEL,
    PATH_CHECKPOINT, 
    DEVICE,
    MODEL_STATE_DICT,
    EVALUATION_STRATEGY,
    METRIC_BEST_MODEL,
    SCHEDULER_TYPE,
    SAVE_STRATEGY,
    NUM_TRAIN_EPOCHS,
    SEED,
    DATA_SEED,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SAVE_TOTAL_LIMIT,
    EARLY_STOPPING_PATIENCE,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE
)

from transformers import (
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback,
    BertForSequenceClassification
)

def fine_tuning_model_and_save(model, train_dataset, validation_dataset):
    """
    Train and fine-tune the model with the specified training arguments.

    Parameters:
    path_save_model (str): Path to save the final trained model.
    path_checkpoint (str): Path to save training checkpoints.
    gpu (bool): Flag to indicate if GPU is used. Defaults to True.

    Returns:
    Trainer: The trained model's trainer instance.
    """
    # Define training arguments
    training_args = TrainingArguments(
        output_dir = PATH_CHECKPOINT,
        per_device_train_batch_size = TRAIN_BATCH_SIZE,
        per_device_eval_batch_size = EVAL_BATCH_SIZE,
        evaluation_strategy = EVALUATION_STRATEGY,
        save_strategy = SAVE_STRATEGY,
        do_train = True,
        do_eval = True,
        num_train_epochs = NUM_TRAIN_EPOCHS,
        overwrite_output_dir = True,
        save_total_limit = SAVE_TOTAL_LIMIT,
        fp16 = True,
        seed = SEED,
        data_seed = DATA_SEED,
        metric_for_best_model = METRIC_BEST_MODEL,
        load_best_model_at_end=True,
        learning_rate = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
        lr_scheduler_type = SCHEDULER_TYPE
    )

    # Define trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = validation_dataset,
        compute_metrics = compute_accuracy_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience = EARLY_STOPPING_PATIENCE)]
    )

    # Train the model
    trainer.train()

    # Save the model
    torch.save({'model_state_dict': model.state_dict(), 'training_args': training_args}, PATH_SAVE_MODEL)

    return trainer


def load_trained_model():
    """
    Load a trained BERT model for sequence classification.

    Parameters:
    path_model_hf (str): Path to the pre-trained model from Huggingface.
    path_save_model (str): Path to the saved model checkpoint.
    device (str): Device to load the model onto ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
    BertForSequenceClassification: The loaded BERT model.
    """
    # Load the pre-trained BERT model with sequence classification head
    model = BertForSequenceClassification.from_pretrained(
        PATH_MODEL_HF,
        return_dict = True,
        num_labels = 2,
        output_hidden_states = True
    )

    # Load the model checkpoint
    checkpoint = torch.load(PATH_SAVE_MODEL, map_location = torch.device(DEVICE))

    # Load the state dictionary into the model
    model.load_state_dict(checkpoint[MODEL_STATE_DICT])

    # Move the model to the specified device
    model.to(DEVICE)

    return model