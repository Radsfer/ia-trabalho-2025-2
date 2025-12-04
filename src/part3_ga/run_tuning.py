import sys
import os
import time
import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Setup de Paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.common.seeds import set_seeds, DEFAULT_SEED
from src.common.utils_io import ensure_folder
from src.part3_ga.ga import GeneticAlgorithm

# Caminhos
PROCESSED_PATH = "data/processed/"
REPORTS_DIR = "reports/part3_ga/"

def load_data_subset(size=2000):
    try:
        X_train = np.load(os.path.join(PROCESSED_PATH, 'X_train.npy'))
        y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train.npy'))
        np.random.seed(DEFAULT_SEED)
        indices = np.random.choice(len(X_train), size=min(size, len(X_train)), replace=False)
        return X_train[indices], y_train[indices]
    except FileNotFoundError:
        print("Erro: Arquivos .npy não encontrados. Rode 'make part2' primeiro.")
        sys.exit(1)

X_sub, y_sub = None, None

def fitness_svm(genes):
    global X_sub, y_sub
    C_val, gamma_val = genes[0], genes[1]
    if C_val <= 0: C_val = 0.001
    if gamma_val <= 0: gamma_val = 0.0001
        
    model = SVC(C=C_val, gamma=gamma_val, kernel='rbf', class_weight='balanced', random_state=DEFAULT_SEED)
    split = int(len(X_sub) * 0.8)
    X_t, X_v = X_sub[:split], X_sub[split:]
    y_t, y_v = y_sub[:split], y_sub[split:]
    model.fit(X_t, y_t)
    return accuracy_score(y_v, model.predict(X_v))

def save_history_csv(history):
    """Salva a evolução da acurácia geração por geração."""
    csv_path = os.path.join(REPORTS_DIR, "generation_history.csv")
    ensure_folder(csv_path)
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Best_Fitness"])
        for gen, fit in enumerate(history, start=1):
            writer.writerow([gen, f"{fit:.6f}"])
            
    print(f"✅ Histórico de Gerações salvo em: {csv_path}")

def save_final_results_csv(best_genes, best_acc, elapsed_time):
    csv_path = os.path.join(REPORTS_DIR, "tuning_results.csv")
    ensure_folder(csv_path)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Best_C", "Best_Gamma", "Accuracy", "Time_Seconds"])
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            f"{best_genes[0]:.6f}",
            f"{best_genes[1]:.6f}",
            f"{best_acc:.4f}",
            f"{elapsed_time:.2f}"
        ])
    print(f"✅ Resultado Final salvo em: {csv_path}")

def main():
    global X_sub, y_sub
    set_seeds()
    print(">>> [AG] Carregando dados...")
    X_sub, y_sub = load_data_subset(size=2000)
    
    bounds = [(0.1, 100.0), (0.0001, 1.0)]
    generations = 10
    
    ga = GeneticAlgorithm(
        pop_size=10,
        mutation_rate=0.2,
        crossover_rate=0.8,
        elitism_count=2,
        fitness_func=fitness_svm,
        gene_bounds=bounds
    )
    
    start_time = time.time()
    # Recebe o histórico aqui
    best_genes, best_acc, history = ga.run(generations=generations)
    end_time = time.time()
    
    print("\n" + "="*40)
    print(f"RESULTADO FINAL (Tempo: {end_time - start_time:.2f}s)")
    print(f"  C={best_genes[0]:.4f}, Gamma={best_genes[1]:.4f}")
    print(f"  Acurácia={best_acc:.4f}")
    print("="*40)
    
    # Salva os dois arquivos
    save_history_csv(history)
    save_final_results_csv(best_genes, best_acc, end_time - start_time)

if __name__ == "__main__":
    main()