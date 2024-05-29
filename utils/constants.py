import torch

#----------------------------------------------------------------------------------------------------------------------------------------------#
# --------------------------------------------- Paths de Google Drive a utilizar --------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------------#
PATH_DATASET_TRAINING = './drive/MyDrive/DATASETS_FINANCIAL_TEXT_SIMPLIFICATION/Uncertainty_Datasets/Datasets/Datasets_Training/'
PATH_DATASET_TEST = './drive/MyDrive/DATASETS_FINANCIAL_TEXT_SIMPLIFICATION/Uncertainty_Datasets/Datasets/Datasets_Test_OOD/'
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