import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils import (
    BATCH_SIZE,  
    PATH_DATASET_TEST, 
    PATH_DATASET_TRAINING,
    NAME_COLUMN_SIMPLE_MANUAL
)

class Datasets:
    _instance = None


    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Datasets, cls).__new__(cls)
        return cls._instance
    
    
    def __init__(self):
        if not hasattr(self, "_instance"):
            self.path_dataset_test = PATH_DATASET_TEST
            self.path_dataset_train = PATH_DATASET_TRAINING
            self.batch_size = BATCH_SIZE
            self._initialized = True
            

    def __prepare_dataset(dataset, name_col_complex, name_col_simple, tipo, column=1):
        """
        Prepare the final dataset for training or testing.

        Parameters:
        dataset (pd.DataFrame): The dataset to prepare.
        name_col_complex (str): Name of the column containing complex segments.
        name_col_simple (str): Name of the column containing simple segments.
        tipo (str): Type of dataset preparation ('general' or 'especifico').
        column (str or int): Column to analyze only in 'especifico' type.

        Returns:
        pd.DataFrame: The prepared dataset with 'label' and 'text' columns.
        """
        if tipo == 'general':
            labels = pd.DataFrame(np.repeat(np.array([0, 1]), len(dataset)))
            texts = pd.concat([dataset[name_col_complex], dataset[name_col_simple]], ignore_index=True)
            prepared_dataset = pd.concat([labels, texts], axis=1)
            prepared_dataset.columns = ['label', 'text']
        elif tipo == 'especifico':
            prepared_dataset = pd.concat([dataset.iloc[:, column], dataset[name_col_complex]], axis=1)
            prepared_dataset.columns = ['label', 'text']
            prepared_dataset['label'] = prepared_dataset['label'].astype(int)
        else:
            raise ValueError("Invalid 'tipo' value. Use 'general' or 'especifico'.")

        return prepared_dataset


    def __split_dataset(data, perc, seed):
        """
        Split dataset into training and validation sets based on a given percentage.

        Parameters:
        data (pd.DataFrame): The dataset to split.
        perc (float): The percentage of data to use for validation.
        seed (int): The seed for random number generation.

        Returns:
        tuple: Two DataFrames, one for training and one for validation.
        """
        np.random.seed(seed)
        random_indices = np.random.rand(len(data))
        
        train_data = data[random_indices >= perc]
        val_data = data[random_indices < perc]

        return train_data, val_data


    def load_datasets(self):
        """
        Load and preprocess training and testing datasets from given directories.

        Returns:
        tuple: Three lists containing the training, validation, and testing datasets respectively.
        """
        # Load names of documents
        files_tr = sorted(os.listdir(self.path_dataset_train))
        files_te = sorted(os.listdir(self.path_dataset_test))

        # Initialize lists to hold datasets
        train_datasets = []
        val_datasets = []
        test_datasets = []

        # Load files and process datasets
        for train_file, test_file in zip(files_tr, files_te):
            
            train_temp = pd.read_csv(os.path.join(self.path_dataset_train, train_file))
            test_temp = pd.read_csv(os.path.join(self.path_dataset_test, test_file))

            train_temp, val_temp = self.__split_dataset(train_temp, perc=test_temp.shape[0] / train_temp.shape[0], seed=20)

            train_temp = self.__prepare_dataset(train_temp, name_col_complex='Segmento', name_col_simple=NAME_COLUMN_SIMPLE_MANUAL, tipo='general', column=1)
            val_temp = self.__prepare_dataset(val_temp, name_col_complex='Segmento', name_col_simple=NAME_COLUMN_SIMPLE_MANUAL, tipo='general', column=1)
            test_temp = self.__prepare_dataset(test_temp, name_col_complex='Segmento', name_col_simple=NAME_COLUMN_SIMPLE_MANUAL, tipo='general', column=1)

            train_datasets.append(train_temp)
            val_datasets.append(val_temp)
            test_datasets.append(test_temp)

        return train_datasets, val_datasets, test_datasets


    def get_dataloader_by_dataset(self, dataset):
        """
        Create DataLoaders for tokenized dataset components.

        Parameters:
        dataset (Dataset): Tokenized dataset containing "labels", 'token_type_ids', 'input_ids', and 'attention_mask'.

        Returns:
        tuple: DataLoaders for 'input_ids', 'attention_mask', and 'token_type_ids'.
        """
        # Get the labels from the 'test_es' dataset and store them in a NumPy array
        labels = DataLoader(torch.tensor(dataset["label"]), batch_size = self.batch_size)  
        
        # Get token_type_ids and create DataLoader
        token_type_ids = DataLoader(torch.tensor(dataset['token_type_ids']), batch_size = self.batch_size)

        # Get input_ids and create DataLoader
        input_ids = DataLoader(torch.tensor(dataset['input_ids']), batch_size = self.batch_size)

        # Get attention_mask and create DataLoader
        attention_mask = DataLoader(torch.tensor(dataset['attention_mask']), batch_size = self.batch_size)

        return labels, input_ids, attention_mask, token_type_ids

