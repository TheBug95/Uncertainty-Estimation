import pyarrow as pa
from datasets import Dataset

def tokenize_datasets(tokenizer, data, index_data, batch_size):
    """
    Tokenize and batch datasets for training, validation, and testing.

    Parameters:
    tokenizer (PreTrainedTokenizer): The tokenizer to use for text tokenization.
    tr (list): List of training datasets.
    va (list): List of validation datasets.
    te (list): List of testing datasets.
    index_data (int): Index to select a specific dataset from the lists.
    batch_size (int): Size of the batches for tokenization.

    Returns:
    tuple: Tokenized and batched datasets for training, validation, and testing.
    """
    # Select the specific fold to work with
    data_selected = data[index_data]

    # Convert to Huggingface datasets
    dataset = Dataset(pa.Table.from_pandas(data_selected))

    # Tokenize text and batch the datasets
    return dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        batch_size=batch_size,
        remove_columns=['text']
    )

     
def tokenize_function(examples, tokenizer):
    """
    Tokenize text data in the examples using the given tokenizer.

    Parameters:
    examples (dict): A dictionary containing text data to tokenize.
    tokenizer (PreTrainedTokenizer): The tokenizer to use for text tokenization.

    Returns:
    dict: A dictionary with tokenized text data.
    """
    max_length = 150
    return tokenizer(examples["text"], padding = "max_length", truncation = True, max_length = max_length, return_tensors = "pt")