import sys
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.common.seeds import set_seeds, DEFAULT_SEED
from src.part2_ml.utils_metrics import evaluate_model

PROCESSED_PATH = "data/processed/"

def main():
    set_seeds()
    
    print(">>> [TREE] Carregando dados...")
    try:
        X_train = np.load(os.path.join(PROCESSED_PATH, 'X_train.npy'))
        y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train.npy'))
        X_test = np.load(os.path.join(PROCESSED_PATH, 'X_test.npy'))
        y_test = np.load(os.path.join(PROCESSED_PATH, 'y_test.npy'))
    except FileNotFoundError:
        return

    print(">>> [TREE] Treinando Árvore de Decisão...")
    model = DecisionTreeClassifier(max_depth=10, random_state=DEFAULT_SEED)
    model.fit(X_train, y_train)
    
    print(">>> [TREE] Avaliando...")
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "Decision Tree Classifier")

if __name__ == "__main__":
    main()