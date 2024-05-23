import torch
import gc

def clean_gpu():    
    print("Memoria GPU reservada antes de limpiar: ")
    print(torch.cuda.memory_reserved())
    gc.collect()
    torch.cuda.empty_cache()
    print("Memoria GPU reservada despues de limpiar: ")
    print(torch.cuda.memory_reserved())