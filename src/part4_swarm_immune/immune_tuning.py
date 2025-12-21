import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Configuração de Caminhos ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.common.seeds import set_seeds, DEFAULT_SEED

PROCESSED_PATH = "data/processed/"
REPORTS_PATH = "reports/part4_swarm/" # Salvando no mesmo lugar para facilitar
os.makedirs(REPORTS_PATH, exist_ok=True)

# --- 1. Classe CLONALG (Sistema Imune) ---
class CLONALG:
    def __init__(self, objective_func, bounds, pop_size=10, selection_size=5, clones_per_factor=3, max_iter=5):
        """
        :param pop_size: Tamanho da população de anticorpos
        :param selection_size: Quantos melhores são selecionados para clonagem
        :param clones_per_factor: Fator de multiplicação de clones
        """
        self.func = objective_func
        self.bounds = bounds
        self.pop_size = pop_size
        self.sel_size = selection_size
        self.clones_factor = clones_per_factor
        self.max_iter = max_iter
        
        # População inicial (Lista de anticorpos)
        self.population = []
        for _ in range(pop_size):
            ab = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            self.population.append(ab)
            
        self.best_solution = None
        self.best_score = -np.inf
        self.history = []

    def mutate(self, antibody, rank, total_ranks):
        """
        Hipermutação Somática:
        Anticorpos melhores (rank baixo) mutam MENOS.
        Anticorpos piores mutam MAIS (para tentar sorte).
        """
        mutated = antibody.copy()
        
        # Taxa de mutação baseada no rank (decai exponencialmente para os melhores)
        # Quanto melhor o rank (0 é o melhor), menor a mutação
        mutation_rate = np.exp(-rank / total_ranks) 
        
        for i in range(len(self.bounds)):
            # Aplica ruído gaussiano
            if np.random.random() < 0.8: # 80% de chance de alterar o gene
                r_range = self.bounds[i][1] - self.bounds[i][0]
                noise = np.random.normal(0, r_range * 0.1 * mutation_rate)
                mutated[i] += noise
                # Garante limites
                mutated[i] = np.clip(mutated[i], self.bounds[i][0], self.bounds[i][1])
                
        return mutated

    def optimize(self):
        for i in range(self.max_iter):
            print(f"--- Geração {i+1}/{self.max_iter} ---")
            
            # 1. Avaliação de Afinidade (Fitness)
            pop_scores = []
            for ab in self.population:
                score = self.func(ab)
                pop_scores.append((ab, score))
                
                # Salvar Histórico
                self.history.append({
                    'generation': i + 1,
                    'n_estimators': int(ab[0]),
                    'max_depth': int(ab[1]),
                    'score': score
                })
            
            # Ordenar do melhor para o pior
            pop_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Atualiza o melhor global
            current_best_ab, current_best_val = pop_scores[0]
            if current_best_val > self.best_score:
                self.best_score = current_best_val
                self.best_solution = current_best_ab.copy()
                print(f"  🦠 Novo Melhor Anticorpo: {self.best_score:.4f} (Params: {np.round(self.best_solution).astype(int)})")
            
            # 2. Seleção Clonal (Pega os Top-N)
            selected = pop_scores[:self.sel_size]
            
            # 3. Clonagem e Maturação (Mutação)
            clones_pop = []
            for rank, (ab, fit) in enumerate(selected):
                # Quantidade de clones pode ser proporcional ao rank, mas fixo simplifica
                n_clones = self.clones_factor 
                
                for _ in range(n_clones):
                    mutated_clone = self.mutate(ab, rank, self.sel_size)
                    clones_pop.append(mutated_clone)
            
            # Avaliar os clones
            # (Adicionamos os clones à população da próxima geração)
            # Estratégia CLONALG padrão: Substitui os piores pelos novos clones se forem melhores
            # Simplificação: Nova população = Melhores Originais + Clones Mutados + Novos Aleatórios (Diversidade)
            
            new_pop = [p[0] for p in selected] # Mantém a elite (Top N)
            
            # Adiciona os clones (apenas os anticorpos, sem score ainda, serão avaliados na prox iteração)
            for c in clones_pop:
                new_pop.append(c)
            
            # Preenche o resto com aleatórios (Receptor Editing - Diversidade)
            while len(new_pop) < self.pop_size:
                ab = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])
                new_pop.append(ab)
            
            # Corta excesso se houver
            self.population = new_pop[:self.pop_size]

        return self.best_solution, self.best_score, pd.DataFrame(self.history)

# --- 2. Carregamento de Dados ---
def load_data():
    try:
        X_train = np.load(os.path.join(PROCESSED_PATH, 'X_train.npy'))
        y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train.npy'))
        X_test = np.load(os.path.join(PROCESSED_PATH, 'X_test.npy'))
        y_test = np.load(os.path.join(PROCESSED_PATH, 'y_test.npy'))
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        print(f"❌ Erro: Arquivos .npy não encontrados em {PROCESSED_PATH}")
        sys.exit(1)

X_train_global, y_train_global = None, None

# --- 3. Função Objetivo ---
def objective_function(params):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    if n_estimators < 1: n_estimators = 1
    if max_depth < 1: max_depth = 1
    
    # Random Forest (Mesmo modelo do PSO para comparação justa)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=DEFAULT_SEED, n_jobs=-1)
    scores = cross_val_score(clf, X_train_global, y_train_global, cv=3, scoring='accuracy')
    return scores.mean()

# --- 4. Execução ---
if __name__ == "__main__":
    set_seeds()
    print(">>> [CLONALG] Carregando dados...")
    X_train_global, y_train_global, X_test, y_test = load_data()
    
    # Configuração igual ao PSO para comparação
    bounds = [(10, 100), (2, 20)] 
    
    # pop_size=10, selection=5, clones=2 -> Gera ~10 clones por geração + diversidade
    immune = CLONALG(objective_function, bounds, pop_size=10, selection_size=5, clones_per_factor=2, max_iter=5)
    
    print(">>> Iniciando Sistema Imune...")
    best_params, best_cv_score, df_history = immune.optimize()
    
    # Parâmetros Finais
    best_n = int(best_params[0])
    best_d = int(best_params[1])
    
    # --- Validação Final ---
    print("\n>>> 🧪 Validando Modelo Final (Teste Cego)...")
    final_model = RandomForestClassifier(n_estimators=best_n, max_depth=best_d, random_state=DEFAULT_SEED)
    final_model.fit(X_train_global, y_train_global)
    test_acc = accuracy_score(y_test, final_model.predict(X_test))
    
    # --- Relatórios ---
    # 1. CSV
    csv_path = os.path.join(REPORTS_PATH, "immune_history.csv")
    df_history.to_csv(csv_path, index=False)
    
    # 2. Gráfico
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_history, x='generation', y='score', label='Anticorpos')
    best_per_gen = df_history.groupby('generation')['score'].max()
    plt.plot(best_per_gen.index, best_per_gen.values, 'r--', linewidth=2, label='Melhor Afinidade')
    plt.title('Evolução do CLONALG (Sistema Imune)')
    plt.xlabel('Geração')
    plt.ylabel('Acurácia (CV)')
    plt.grid(True)
    plt.savefig(os.path.join(REPORTS_PATH, "immune_convergence.png"))
    
    # 3. Resumo
    report = f"""
    === RESULTADOS CLONALG (IMUNE) ===
    Melhores Params: n_estimators={best_n}, max_depth={best_d}
    Acurácia Treino (CV): {best_cv_score:.4f}
    Acurácia Teste:       {test_acc:.4f}
    """
    print(report)
    with open(os.path.join(REPORTS_PATH, "immune_final_result.txt"), "w") as f:
        f.write(report)