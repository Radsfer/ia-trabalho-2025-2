import sys
import os
import numpy as np
import pytest

# Adiciona pasta src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Testa se as sementes estão fixas
from common.seeds import set_seeds, DEFAULT_SEED

def test_seed_consistency():
    set_seeds()
    a = np.random.rand()
    set_seeds()
    b = np.random.rand()
    assert a == b, "A semente não está garantindo reprodutibilidade!"

# Testa se os arquivos de dados existem
def test_data_files_exist():
    required_files = [
        'data/processed/X_train.npy',
        'data/processed/y_train.npy'
    ]
    for f in required_files:
        assert os.path.exists(f), f"Arquivo {f} não encontrado. Rode o preprocess.py primeiro."