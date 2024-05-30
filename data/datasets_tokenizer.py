import pyarrow as pa
from datasets import Dataset
from transformers import BertTokenizer
from utils import ( 
    BATCH_SIZE, 
    INDEX_DOCUMENT_DATA, 
    MAX_NUM_TOKENS,
    PATH_MODEL_HF
)

class Tokenizer:
    
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Tokenizer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.tokenizer = BertTokenizer.from_pretrained(PATH_MODEL_HF, do_lower_case = False)
            self._initialized = True

    def __tokenize_function(examples, tokenizer):
        """
        Tokenize text data in the examples using the given tokenizer.

        Parameters:
        examples (dict): A dictionary containing text data to tokenize.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for text tokenization.

        Returns:
        dict: A dictionary with tokenized text data.
        """
        return tokenizer(examples["text"], padding = "max_length", truncation = True, max_length = MAX_NUM_TOKENS, return_tensors = "pt")
    
    def tokenize_datasets(self, data):
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
        data_selected = data[INDEX_DOCUMENT_DATA]

        # Convert to Huggingface datasets
        dataset = Dataset(pa.Table.from_pandas(data_selected))

        # Tokenize text and batch the datasets
        return dataset.map(
            lambda examples: self.__tokenize_function(examples, self.tokenizer),
            batched = True,
            batch_size = BATCH_SIZE,
            remove_columns=['text']
        )