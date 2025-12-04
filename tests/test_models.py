import os
import sys
import pytest
import numpy as np
import pandas as pd

# Adiciona o diretório src ao sys.path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# -------------------------------------------------------------------------
# CONSTANTES E CAMINHOS
# -------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "part2_ml")

# -------------------------------------------------------------------------
# TESTES DE DADOS (PREPROCESSAMENTO)
# -------------------------------------------------------------------------

def test_processed_files_exist():
    """Verifica se o script preprocess.py gerou os 4 arquivos obrigatórios."""
    required_files = [
        "X_train.npy", "X_test.npy", 
        "y_train.npy", "y_test.npy"
    ]
    for f in required_files:
        path = os.path.join(DATA_PROC_DIR, f)
        assert os.path.exists(path), f"Arquivo de dados faltando: {f}. Rode 'make part2' antes."

def test_data_shapes_consistency():
    """Verifica se X e y têm o mesmo número de linhas (não perdemos dados)."""
    try:
        X_train = np.load(os.path.join(DATA_PROC_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(DATA_PROC_DIR, "y_train.npy"))
        X_test = np.load(os.path.join(DATA_PROC_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(DATA_PROC_DIR, "y_test.npy"))
    except FileNotFoundError:
        pytest.fail("Dados não encontrados. Rode 'make preprocess' primeiro.")

    # O número de linhas (amostras) deve bater
    assert X_train.shape[0] == y_train.shape[0], "Erro: X_train e y_train com tamanhos diferentes!"
    assert X_test.shape[0] == y_test.shape[0], "Erro: X_test e y_test com tamanhos diferentes!"
    
    # Não pode estar vazio
    assert X_train.shape[0] > 0, "O dataset de treino está vazio!"

# -------------------------------------------------------------------------
# TESTES DE INTEGRAÇÃO (MODELOS RODARAM?)
# -------------------------------------------------------------------------

def test_metrics_report_generated():
    """Verifica se os scripts de treino geraram o CSV de métricas."""
    csv_path = os.path.join(REPORTS_DIR, "metrics.csv")
    assert os.path.exists(csv_path), "O arquivo metrics.csv não existe. Os modelos rodaram?"

def test_all_models_are_present():
    """Lê o CSV e verifica se KNN, SVM e Árvore estão lá."""
    csv_path = os.path.join(REPORTS_DIR, "metrics.csv")
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        pytest.fail("Não foi possível ler metrics.csv")

    # Verifica se os nomes dos modelos aparecem na coluna 'Model'
    # Nota: Ajuste a string se tiver mudado o nome no utils_metrics.py
    models_found = df['Model'].tolist()
    expected_models = ["KNN", "SVM", "Decision Tree"]
    
    for expected in expected_models:
        # Verifica se alguma string da coluna contem o nome esperado (ex: "K-Nearest Neighbors (KNN)")
        assert any(expected in m for m in models_found), f"O modelo {expected} não aparece no relatório final!"

def test_sanity_check_accuracy():
    """Teste de Sanidade: A acurácia não pode ser zero (o modelo aprendeu algo?)."""
    csv_path = os.path.join(REPORTS_DIR, "metrics.csv")
    df = pd.read_csv(csv_path)
    
    # Garante que a acurácia é um número válido maior que 0.1 (chute aleatório seria 0.5 no balanceado)
    # Se der 0.0, tem algo muito errado no código.
    assert (df['Accuracy'] > 0.1).all(), "Algum modelo teve acurácia suspeita (<= 10%). Verifique o treino."