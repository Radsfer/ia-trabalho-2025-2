import sys
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Imports locais
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.common.seeds import set_seeds
from src.part2_ml.utils_metrics import evaluate_model

PROCESSED_PATH = "data/processed/"

def main():
    set_seeds()
    
    print(">>> [KNN] Carregando dados...")
    try:
        X_train = np.load(os.path.join(PROCESSED_PATH, 'X_train.npy'))
        y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train.npy'))
        X_test = np.load(os.path.join(PROCESSED_PATH, 'X_test.npy'))
        y_test = np.load(os.path.join(PROCESSED_PATH, 'y_test.npy'))
    except FileNotFoundError:
        print("Erro: Execute 'preprocess.py' antes!")
        return

    print(">>> [KNN] Treinando modelo (k=5)...")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    
    print(">>> [KNN] Avaliando...")
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "K-Nearest Neighbors (KNN)")

if __name__ == "__main__":
    main()