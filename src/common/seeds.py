import os
import random
import numpy as np

# Defina a semente global aqui
DEFAULT_SEED = 42

def set_seeds(seed=DEFAULT_SEED):
    """Configura as sementes para garantir reprodutibilidade."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Se fosse usar torch ou tensorflow, configuraria aqui também
    
    return seed