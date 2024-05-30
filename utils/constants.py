import torch

#----------------------------------------------------------------------------------------------------------------------------------------------#
# --------------------------------------------- Paths de Google Drive a utilizar --------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------------#
PATH_DATASET_TRAINING = './drive/MyDrive/DATASETS_FINANCIAL_TEXT_SIMPLIFICATION/Uncertainty_Datasets/Datasets/Datasets_Training/'
PATH_DATASET_TEST = './drive/MyDrive/DATASETS_FINANCIAL_TEXT_SIMPLIFICATION/Uncertainty_Datasets/Datasets/Datasets_Test/'
PATH_SAVE_MODEL = "./drive/MyDrive/DATASETS_FINANCIAL_TEXT_SIMPLIFICATION/Uncertainty_Datasets/Datasets/Model_Trained/model_trained.bin"
PATH_CHECKPOINT = "./drive/MyDrive/DATASETS_FINANCIAL_TEXT_SIMPLIFICATION/Uncertainty_Datasets/Datasets/Model_Check_Point/"
PATH_LOGGER = '/content/drive/MyDrive/DATASETS_FINANCIAL_TEXT_SIMPLIFICATION/Registro_Eventos/log.txt'

#----------------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------- Path para cargar el modelo de Hugging Face -------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------------#
PATH_MODEL_HF = 'dccuchile/bert-base-spanish-wwm-uncased'

#----------------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------- Establecer dispositivo de procesamiento por default -------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------------#
DEVICE, GPU = ((torch.device(("cuda:0")), True) if torch.cuda.is_available() else ("cpu", False))

#----------------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------- Capa a extraer las características y tamaño de batch ------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------------#
LAYER_INDEX = 12
BATCH_SIZE = 10
INDEX_DOCUMENT_DATA = 0

#----------------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------- Arguments for finetuning models ------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------------#
MODEL_STATE_DICT = "model_state_dict"
EVALUATION_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
METRIC_BEST_MODEL = "accuracy"
SCHEDULER_TYPE = 'linear'
NUM_TRAIN_EPOCHS = 6
SEED = 65
DATA_SEED = 90
LEARNING_RATE = 3.419313942464226e-05
WEIGHT_DECAY = 0.4
SAVE_TOTAL_LIMIT = 1
EARLY_STOPPING_PATIENCE = 2
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

#----------------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------- Tokenizer for BERT Model ------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------------#

NAME_COLUMN_SIMPLE_MANUAL = 'simpleManual'
MAX_NUM_TOKENS = 150