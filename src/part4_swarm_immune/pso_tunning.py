import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Configuracao de caminhos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.common.seeds import set_seeds, DEFAULT_SEED

PROCESSED_PATH = "data/processed/"
REPORTS_PATH = "reports/part4_swarm_immune/"

# Cria pasta de reports se nao existir
os.makedirs(REPORTS_PATH, exist_ok=True)

# --- 1. Configuracao do PSO ---
class Particle:
    def __init__(self, bounds):
        self.position = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        self.velocity = np.array([np.random.uniform(-1, 1) for _ in bounds])
        self.best_position = self.position.copy()
        self.best_score = -np.inf
        self.score = -np.inf

class PSO:
    def __init__(self, objective_func, bounds, n_particles=10, max_iter=10):
        self.func = objective_func
        self.bounds = bounds
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.global_best_position = None
        self.global_best_score = -np.inf
        self.particles = []
        self.history = []
        
        self.w = 0.5  # Inercia
        self.c1 = 1.5 # Cognitivo
        self.c2 = 1.5 # Social

    def optimize(self):
        self.particles = [Particle(self.bounds) for _ in range(self.n_particles)]
        
        for i in range(self.max_iter):
            print(f"--- Iteracao {i+1}/{self.max_iter} ---")
            
            for particle in self.particles:
                params = particle.position
                score = self.func(params)
                
                # Salvar no historico
                self.history.append({
                    'iteration': i + 1,
                    'C': params[0],
                    'gamma': params[1],
                    'score': score
                })
                
                # Atualizar PBest
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()
                
                # Atualizar GBest
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle.position.copy()
                    print(f"  Novo gBest: {score:.4f} (C={params[0]:.4f}, gamma={params[1]:.6f})")

            # Mover Particulas
            for particle in self.particles:
                r1, r2 = np.random.rand(), np.random.rand()
                particle.velocity = (self.w * particle.velocity) + \
                                    (self.c1 * r1 * (particle.best_position - particle.position)) + \
                                    (self.c2 * r2 * (self.global_best_position - particle.position))
                particle.position += particle.velocity
                for j in range(len(self.bounds)):
                    particle.position[j] = np.clip(particle.position[j], self.bounds[j][0], self.bounds[j][1])
        
        return self.global_best_position, self.global_best_score, pd.DataFrame(self.history)

# --- 2. Carregamento (Subset para comparacao justa com AG) ---
def load_data_subset(size=2000):
    """Carrega subset de dados igual ao AG da Parte 3."""
    try:
        X_train = np.load(os.path.join(PROCESSED_PATH, 'X_train.npy'))
        y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train.npy'))
        X_test = np.load(os.path.join(PROCESSED_PATH, 'X_test.npy'))
        y_test = np.load(os.path.join(PROCESSED_PATH, 'y_test.npy'))
        
        # Subset para treino (mesmo tamanho do AG)
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
    
    # Protecao contra valores invalidos
    if C_val <= 0: C_val = 0.001
    if gamma_val <= 0: gamma_val = 0.0001
    
    # SVM com kernel RBF (mesmo da Parte 3)
    model = SVC(C=C_val, gamma=gamma_val, kernel='rbf', class_weight='balanced', random_state=DEFAULT_SEED)
    
    # Split 80/20 (mesmo metodo do AG para comparacao justa)
    split = int(len(X_sub) * 0.8)
    X_t, X_v = X_sub[:split], X_sub[split:]
    y_t, y_v = y_sub[:split], y_sub[split:]
    
    model.fit(X_t, y_t)
    return accuracy_score(y_v, model.predict(X_v))

# --- 4. Execucao ---
if __name__ == "__main__":
    set_seeds()
    print(">>> [PSO] Carregando dados (subset 2000)...")
    X_sub, y_sub, X_test, y_test = load_data_subset(size=2000)
    
    print(f">>> Subset Treino: {X_sub.shape}, Teste: {X_test.shape}")
    
    # Bounds identicos a Parte 3: C [0.1, 100], Gamma [0.0001, 1.0]
    bounds = [(0.1, 100.0), (0.0001, 1.0)]
    pso = PSO(objective_function, bounds, n_particles=10, max_iter=10)
    
    print(">>> Iniciando Otimizacao PSO...")
    best_params_raw, best_cv_score, df_history = pso.optimize()
    
    # Parametros finais
    best_C = best_params_raw[0]
    best_gamma = best_params_raw[1]

    # --- 5. Validacao Final no Teste Cego ---
    print("\n>>> Validando Modelo Final no Conjunto de Teste...")
    
    # Treina no subset completo, testa no conjunto de teste real
    final_model = SVC(C=best_C, gamma=best_gamma, kernel='rbf', class_weight='balanced', random_state=DEFAULT_SEED)
    final_model.fit(X_sub, y_sub)
    y_pred = final_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    # --- 6. Salvar Resultados ---
    
    # 6.1 CSV de historico
    csv_path = os.path.join(REPORTS_PATH, "pso_history.csv")
    df_history.to_csv(csv_path, index=False)
    print(f"Historico salvo em: {csv_path}")

    # 6.2 Grafico de Convergencia
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_history, x='iteration', y='score', marker='o', label='Particulas')
    best_per_iter = df_history.groupby('iteration')['score'].max()
    plt.plot(best_per_iter.index, best_per_iter.values, 'r--', linewidth=2, label='Melhor Global')
    plt.title('Convergencia do PSO - Otimizacao SVM')
    plt.xlabel('Iteracao')
    plt.ylabel('Acuracia')
    plt.legend()
    plt.grid(True)
    img_path = "reports/figs/pso_convergence.png"
    plt.savefig(img_path)
    plt.close()
    print(f"Grafico salvo em: {img_path}")

    # 6.3 Relatorio Final
    report = f"""
    === RESULTADOS PSO (OTIMIZANDO SVM) ===
    Melhores Params: C={best_C:.4f}, gamma={best_gamma:.6f}
    Acuracia Treino (Validacao): {best_cv_score:.4f}
    Acuracia Teste (Cego): {test_acc:.4f}
    
    Diferenca (Treino - Teste): {(best_cv_score - test_acc):.4f}
    """
    print(report)
    with open(os.path.join(REPORTS_PATH, "pso_final_result.txt"), "w") as f:
        f.write(report)