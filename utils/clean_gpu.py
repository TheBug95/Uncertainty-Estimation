import torch
import gc
from logger import logger   

def clean_gpu():    
    logger.info(f"Memoria GPU reservada antes de limpiar: {torch.cuda.memory_reserved()}")
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"Memoria GPU reservada después de limpiar: {torch.cuda.memory_reserved()}")