import torch

from utils import compute_metrics
from transformers import TrainingArguments
from transformers import Trainer
from transformers import EarlyStoppingCallback



def fine_tuning_model_and_save(path_save_model, path_checkpoint, model, train_dataset, validation_dataset, gpu=True):
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
        output_dir=path_checkpoint,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        do_train=True,
        do_eval=True,
        num_train_epochs=6,
        overwrite_output_dir=True,
        save_total_limit=1,
        fp16=gpu,
        seed=65,
        data_seed=90,
        metric_for_best_model='accuracy',
        load_best_model_at_end=True,
        learning_rate=3.419313942464226e-05,
        weight_decay=0.4,
        lr_scheduler_type='linear'
    )

    # Define trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = validation_dataset,
        compute_metrics = compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train the model
    trainer.train()

    # Save the model
    torch.save({'model_state_dict': model.state_dict(), 'training_args': training_args}, path_save_model)

    return trainer