import sys
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Setup de Paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.common.seeds import set_seeds, DEFAULT_SEED
from src.part3_ga.ga import GeneticAlgorithm

# Caminhos
PROCESSED_PATH = "data/processed/"

def load_data_subset(size=2000):
    """Carrega apenas um pedaço dos dados para o AG ser rápido."""
    try:
        X_train = np.load(os.path.join(PROCESSED_PATH, 'X_train.npy'))
        y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train.npy'))
        
        # Pega uma amostra fixa para ser justo em todas as gerações
        # Usamos seed fixa para o fitness ser determinístico
        np.random.seed(DEFAULT_SEED)
        indices = np.random.choice(len(X_train), size=min(size, len(X_train)), replace=False)
        
        return X_train[indices], y_train[indices]
    except FileNotFoundError:
        print("Erro: Arquivos .npy não encontrados. Rode 'make part2' primeiro.")
        sys.exit(1)

# Dados globais para o fitness não precisar recarregar toda hora
X_sub, y_sub = None, None

def fitness_svm(genes):
    """
    Recebe [C, Gamma] e retorna a Acurácia.
    genes[0] = C
    genes[1] = Gamma
    """
    global X_sub, y_sub
    
    # Decodifica genes
    c_val = genes[0]
    gamma_val = genes[1]
    
    # Cria modelo (Kernel RBF é o padrão para otimizar C/Gamma)
    # class_weight='balanced' ajuda no nosso problema do Olist
    model = SVC(C=c_val, gamma=gamma_val, kernel='rbf', class_weight='balanced', random_state=DEFAULT_SEED)
    
    # Treina/Valida usando Cross-Validation simples (Holdout 80/20 na amostra)
    # Para ser BEM rápido no AG, vamos fazer um split manual simples
    split = int(len(X_sub) * 0.8)
    X_t, X_v = X_sub[:split], X_sub[split:]
    y_t, y_v = y_sub[:split], y_sub[split:]
    
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    
    return accuracy_score(y_v, preds)

def main():
    global X_sub, y_sub
    set_seeds()
    
    print(">>> [AG] Carregando amostra de dados...")
    X_sub, y_sub = load_data_subset(size=2000) # 2000 linhas para o AG treinar rápido
    
    print(f">>> Amostra carregada: {X_sub.shape}")
    
    # Configuração do AG
    # Genes: [C, Gamma]
    # Limites: C entre 0.1 e 100, Gamma entre 0.0001 e 1.0
    bounds = [(0.1, 100.0), (0.0001, 1.0)]
    
    ga = GeneticAlgorithm(
        pop_size=10,        # 10 indivíduos
        mutation_rate=0.2,  # 20% de chance de mutar
        crossover_rate=0.8, # 80% de chance de cruzar
        elitism_count=2,    # Mantém os 2 melhores sempre
        fitness_func=fitness_svm,
        gene_bounds=bounds
    )
    
    # Rodar 10 gerações (rápido para teste)
    best_genes, best_acc = ga.run(generations=10)
    
    print("\n" + "="*40)
    print(f"RESULTADO FINAL DO AG")
    print("="*40)
    print(f"Melhores Hiperparâmetros encontrados:")
    print(f"  C     = {best_genes[0]:.6f}")
    print(f"  Gamma = {best_genes[1]:.6f}")
    print(f"  Acurácia na Validação = {best_acc:.4f}")
    print("="*40)
    
    # Opcional: Treinar o modelo final no dataset COMPLETO para ver se valeu a pena
    print("\n>>> Validando no Dataset de Teste COMPLETO (original da Parte 2)...")
    try:
        X_train_full = np.load(os.path.join(PROCESSED_PATH, 'X_train.npy'))
        y_train_full = np.load(os.path.join(PROCESSED_PATH, 'y_train.npy'))
        X_test_full = np.load(os.path.join(PROCESSED_PATH, 'X_test.npy'))
        y_test_full = np.load(os.path.join(PROCESSED_PATH, 'y_test.npy'))
        
        final_model = SVC(C=best_genes[0], gamma=best_genes[1], kernel='rbf', class_weight='balanced', random_state=DEFAULT_SEED)
        final_model.fit(X_train_full, y_train_full)
        final_pred = final_model.predict(X_test_full)
        final_acc = accuracy_score(y_test_full, final_pred)
        
        print(f"Acurácia Final (Teste Real): {final_acc:.4f}")
    except Exception as e:
        print(f"Não foi possível validar no full dataset: {e}")

if __name__ == "__main__":
    main()