import os
import numpy as np
import pandas as pd


def prepare_dataset(dataset, name_col_complex, name_col_simple, tipo, column=1):
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


def split_dataset(data, perc, seed):
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


def load_datasets(path_tr, path_te, name_column='simpleManual'):
    """
    Load and preprocess training and testing datasets from given directories.

    Parameters:
    path_tr (str): Path to the directory containing training data files.
    path_te (str): Path to the directory containing testing data files.
    max_length (int): Maximum length of characters for each sentence. Defaults to 150.
    name_column (str): Column name for the simple manual segment. Defaults to 'simpleManual'.

    Returns:
    tuple: Three lists containing the training, validation, and testing datasets respectively.
    """
    # Load names of documents
    files_tr = sorted(os.listdir(path_tr))
    files_te = sorted(os.listdir(path_te))

    # Initialize lists to hold datasets
    train_datasets = []
    val_datasets = []
    test_datasets = []

    # Load files and process datasets
    for train_file, test_file in zip(files_tr, files_te):
        
        train_temp = pd.read_csv(os.path.join(path_tr, train_file))
        test_temp = pd.read_csv(os.path.join(path_te, test_file))

        train_temp, val_temp = split_dataset(train_temp, perc=test_temp.shape[0] / train_temp.shape[0], seed=20)

        train_temp = prepare_dataset(train_temp, name_col_complex='Segmento', name_col_simple=name_column, tipo='general', column=1)
        val_temp = prepare_dataset(val_temp, name_col_complex='Segmento', name_col_simple=name_column, tipo='general', column=1)
        test_temp = prepare_dataset(test_temp, name_col_complex='Segmento', name_col_simple=name_column, tipo='general', column=1)

        train_datasets.append(train_temp)
        val_datasets.append(val_temp)
        test_datasets.append(test_temp)

    return train_datasets, val_datasets, test_datasets