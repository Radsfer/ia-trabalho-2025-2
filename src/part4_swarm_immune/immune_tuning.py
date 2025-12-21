import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --- Configuracao de Caminhos ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.common.seeds import set_seeds, DEFAULT_SEED

PROCESSED_PATH = "data/processed/"
REPORTS_PATH = "reports/part4_swarm_immune/"
os.makedirs(REPORTS_PATH, exist_ok=True)

# --- 1. Classe CLONALG (Sistema Imune) ---
class CLONALG:
    def __init__(self, objective_func, bounds, pop_size=10, selection_size=5, clones_per_factor=3, max_iter=10):
        """
        :param pop_size: Tamanho da populacao de anticorpos
        :param selection_size: Quantos melhores sao selecionados para clonagem
        :param clones_per_factor: Fator de multiplicacao de clones
        """
        self.func = objective_func
        self.bounds = bounds
        self.pop_size = pop_size
        self.sel_size = selection_size
        self.clones_factor = clones_per_factor
        self.max_iter = max_iter
        
        # Populacao inicial (Lista de anticorpos)
        self.population = []
        for _ in range(pop_size):
            ab = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            self.population.append(ab)
            
        self.best_solution = None
        self.best_score = -np.inf
        self.history = []

    def mutate(self, antibody, rank, total_ranks):
        """
        Hipermutacao Somatica:
        Anticorpos melhores (rank baixo) mutam MENOS.
        Anticorpos piores mutam MAIS (para tentar sorte).
        """
        mutated = antibody.copy()
        
        # Taxa de mutacao baseada no rank (decai exponencialmente para os melhores)
        mutation_rate = np.exp(-rank / total_ranks) 
        
        for i in range(len(self.bounds)):
            if np.random.random() < 0.8:  # 80% de chance de alterar o gene
                r_range = self.bounds[i][1] - self.bounds[i][0]
                noise = np.random.normal(0, r_range * 0.1 * mutation_rate)
                mutated[i] += noise
                mutated[i] = np.clip(mutated[i], self.bounds[i][0], self.bounds[i][1])
                
        return mutated

    def optimize(self):
        for i in range(self.max_iter):
            print(f"--- Geracao {i+1}/{self.max_iter} ---")
            
            # 1. Avaliacao de Afinidade (Fitness)
            pop_scores = []
            for ab in self.population:
                score = self.func(ab)
                pop_scores.append((ab, score))
                
                # Salvar Historico
                self.history.append({
                    'generation': i + 1,
                    'C': ab[0],
                    'gamma': ab[1],
                    'score': score
                })
            
            # Ordenar do melhor para o pior
            pop_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Atualiza o melhor global
            current_best_ab, current_best_val = pop_scores[0]
            if current_best_val > self.best_score:
                self.best_score = current_best_val
                self.best_solution = current_best_ab.copy()
                print(f"  Novo Melhor Anticorpo: {self.best_score:.4f} (C={self.best_solution[0]:.4f}, gamma={self.best_solution[1]:.6f})")
            
            # 2. Selecao Clonal (Pega os Top-N)
            selected = pop_scores[:self.sel_size]
            
            # 3. Clonagem e Maturacao (Mutacao)
            clones_pop = []
            for rank, (ab, fit) in enumerate(selected):
                n_clones = self.clones_factor 
                
                for _ in range(n_clones):
                    mutated_clone = self.mutate(ab, rank, self.sel_size)
                    clones_pop.append(mutated_clone)
            
            # Nova populacao = Melhores Originais + Clones Mutados + Novos Aleatorios
            new_pop = [p[0] for p in selected]  # Mantem a elite
            
            for c in clones_pop:
                new_pop.append(c)
            
            # Preenche o resto com aleatorios (Receptor Editing - Diversidade)
            while len(new_pop) < self.pop_size:
                ab = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])
                new_pop.append(ab)
            
            self.population = new_pop[:self.pop_size]

        return self.best_solution, self.best_score, pd.DataFrame(self.history)

# --- 2. Carregamento de Dados (Subset para comparacao justa) ---
def load_data_subset(size=2000):
    """Carrega subset de dados igual ao AG da Parte 3."""
    try:
        X_train = np.load(os.path.join(PROCESSED_PATH, 'X_train.npy'))
        y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train.npy'))
        X_test = np.load(os.path.join(PROCESSED_PATH, 'X_test.npy'))
        y_test = np.load(os.path.join(PROCESSED_PATH, 'y_test.npy'))
        
        np.random.seed(DEFAULT_SEED)
        indices = np.random.choice(len(X_train), size=min(size, len(X_train)), replace=False)
        return X_train[indices], y_train[indices], X_test, y_test
    except FileNotFoundError:
        print("Erro: Arquivos .npy nao encontrados. Rode 'make part2' primeiro.")
        sys.exit(1)

X_sub, y_sub = None, None

# --- 3. Funcao Objetivo (SVM - mesmo da Parte 3) ---
def objective_function(params):
    """Fitness usando SVM para comparacao direta com AG."""
    global X_sub, y_sub
    C_val, gamma_val = params[0], params[1]
    
    if C_val <= 0: C_val = 0.001
    if gamma_val <= 0: gamma_val = 0.0001
    
    model = SVC(C=C_val, gamma=gamma_val, kernel='rbf', class_weight='balanced', random_state=DEFAULT_SEED)
    
    # Split 80/20 (mesmo metodo do AG)
    split = int(len(X_sub) * 0.8)
    X_t, X_v = X_sub[:split], X_sub[split:]
    y_t, y_v = y_sub[:split], y_sub[split:]
    
    model.fit(X_t, y_t)
    return accuracy_score(y_v, model.predict(X_v))

# --- 4. Execucao ---
if __name__ == "__main__":
    set_seeds()
    print(">>> [CLONALG] Carregando dados (subset 2000)...")
    X_sub, y_sub, X_test, y_test = load_data_subset(size=2000)
    
    print(f">>> Subset Treino: {X_sub.shape}, Teste: {X_test.shape}")
    
    # Bounds identicos a Parte 3: C [0.1, 100], Gamma [0.0001, 1.0]
    bounds = [(0.1, 100.0), (0.0001, 1.0)]
    
    # pop_size=10, selection=5, clones=2
    immune = CLONALG(objective_function, bounds, pop_size=10, selection_size=5, clones_per_factor=2, max_iter=10)
    
    print(">>> Iniciando Sistema Imune...")
    best_params, best_cv_score, df_history = immune.optimize()
    
    # Parametros Finais
    best_C = best_params[0]
    best_gamma = best_params[1]
    
    # --- Validacao Final ---
    print("\n>>> Validando Modelo Final (Teste Cego)...")
    final_model = SVC(C=best_C, gamma=best_gamma, kernel='rbf', class_weight='balanced', random_state=DEFAULT_SEED)
    final_model.fit(X_sub, y_sub)
    test_acc = accuracy_score(y_test, final_model.predict(X_test))
    
    # --- Relatorios ---
    # 1. CSV
    csv_path = os.path.join(REPORTS_PATH, "immune_history.csv")
    df_history.to_csv(csv_path, index=False)
    print(f"Historico salvo em: {csv_path}")
    
    # 2. Grafico
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_history, x='generation', y='score', label='Anticorpos')
    best_per_gen = df_history.groupby('generation')['score'].max()
    plt.plot(best_per_gen.index, best_per_gen.values, 'r--', linewidth=2, label='Melhor Afinidade')
    plt.title('Evolucao do CLONALG - Otimizacao SVM')
    plt.xlabel('Geracao')
    plt.ylabel('Acuracia')
    plt.grid(True)
    plt.legend()
    plt.savefig("reports/figs/immune_convergence.png")
    plt.close()
    print("Grafico salvo em: reports/figs/immune_convergence.png")
    
    # 3. Resumo
    report = f"""
    === RESULTADOS CLONALG (OTIMIZANDO SVM) ===
    Melhores Params: C={best_C:.4f}, gamma={best_gamma:.6f}
    Acuracia Treino (Validacao): {best_cv_score:.4f}
    Acuracia Teste (Cego): {test_acc:.4f}
    """
    print(report)
    with open(os.path.join(REPORTS_PATH, "immune_final_result.txt"), "w") as f:
        f.write(report)