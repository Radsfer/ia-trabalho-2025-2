import sys
import os
import numpy as np
from sklearn.svm import SVC

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.common.seeds import set_seeds, DEFAULT_SEED
from src.part2_ml.utils_metrics import evaluate_model

PROCESSED_PATH = "data/processed/"

def main():
    set_seeds()
    
    print(">>> [SVM] Carregando dados...")
    try:
        X_train = np.load(os.path.join(PROCESSED_PATH, 'X_train.npy'))
        y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train.npy'))
        X_test = np.load(os.path.join(PROCESSED_PATH, 'X_test.npy'))
        y_test = np.load(os.path.join(PROCESSED_PATH, 'y_test.npy'))
    except FileNotFoundError:
        return

    print(">>> [SVM] Treinando modelo (pode demorar)...")
    model = SVC(kernel='rbf', C=1.0, random_state=DEFAULT_SEED)
    model.fit(X_train, y_train)
    
    print(">>> [SVM] Avaliando...")
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "Support Vector Machine (SVM)")

if __name__ == "__main__":
    main()