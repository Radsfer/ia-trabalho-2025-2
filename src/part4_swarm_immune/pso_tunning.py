import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import json
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Configuração de caminhos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.common.seeds import set_seeds, DEFAULT_SEED

PROCESSED_PATH = "data/processed/"
REPORTS_PATH = "reports/part4_swarm_immune/"

# Cria pasta de reports se não existir
os.makedirs(REPORTS_PATH, exist_ok=True)

# --- 1. Configuração do PSO ---
class Particle:
    def __init__(self, bounds):
        self.position = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        self.velocity = np.array([np.random.uniform(-1, 1) for _ in bounds])
        self.best_position = self.position.copy()
        self.best_score = -np.inf
        self.score = -np.inf

class PSO:
    def __init__(self, objective_func, bounds, n_particles=10, max_iter=5):
        self.func = objective_func
        self.bounds = bounds
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.global_best_position = None
        self.global_best_score = -np.inf
        self.particles = []
        self.history = [] # Para salvar o log
        
        self.w = 0.5  # Inércia
        self.c1 = 1.5 # Cognitivo
        self.c2 = 1.5 # Social

    def optimize(self):
        self.particles = [Particle(self.bounds) for _ in range(self.n_particles)]
        
        for i in range(self.max_iter):
            print(f"--- Iteração {i+1}/{self.max_iter} ---")
            
            for particle in self.particles:
                params = particle.position
                score = self.func(params)
                
                # Salvar no histórico
                self.history.append({
                    'iteration': i + 1,
                    'n_estimators': int(params[0]),
                    'max_depth': int(params[1]),
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
                    print(f"  🚀 Novo gBest: {score:.4f} (Params: {np.round(particle.position).astype(int)})")

            # Mover Partículas
            for particle in self.particles:
                r1, r2 = np.random.rand(), np.random.rand()
                particle.velocity = (self.w * particle.velocity) + \
                                    (self.c1 * r1 * (particle.best_position - particle.position)) + \
                                    (self.c2 * r2 * (self.global_best_position - particle.position))
                particle.position += particle.velocity
                for j in range(len(self.bounds)):
                    particle.position[j] = np.clip(particle.position[j], self.bounds[j][0], self.bounds[j][1])
        
        return self.global_best_position, self.global_best_score, pd.DataFrame(self.history)

# --- 2. Carregamento ---
def load_data():
    try:
        X_train = np.load(os.path.join(PROCESSED_PATH, 'X_train.npy'))
        y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train.npy'))
        X_test = np.load(os.path.join(PROCESSED_PATH, 'X_test.npy'))
        y_test = np.load(os.path.join(PROCESSED_PATH, 'y_test.npy'))
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        print(f"❌ Erro: Arquivos não encontrados em '{PROCESSED_PATH}'. rode 'python src/part2_ml/preprocess.py'")
        sys.exit(1)

X_train_global, y_train_global = None, None

# --- 3. Função Objetivo ---
def objective_function(params):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    if n_estimators < 1: n_estimators = 1
    if max_depth < 1: max_depth = 1
    
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=DEFAULT_SEED, n_jobs=-1)
    # Validação cruzada evita viés de seleção de dados
    scores = cross_val_score(clf, X_train_global, y_train_global, cv=3, scoring='accuracy')
    return scores.mean()

# --- 4. Execução ---
if __name__ == "__main__":
    set_seeds()
    print(">>> [PSO] Carregando dados...")
    X_train_global, y_train_global, X_test, y_test = load_data()
    
    print(f">>> Treino: {X_train_global.shape}, Teste: {X_test.shape}")
    
    # Configuração do PSO
    bounds = [(10, 100), (2, 20)] # Limites: Arvores [10-100], Profundidade [2-20]
    pso = PSO(objective_function, bounds, n_particles=5, max_iter=5) # Aumente iterações se puder
    
    print(">>> Iniciando Otimização...")
    best_params_raw, best_cv_score, df_history = pso.optimize()
    
    # Arredondar parâmetros finais
    best_n_estimators = int(best_params_raw[0])
    best_max_depth = int(best_params_raw[1])

    # --- 5. Validação Final (A PROVA REAL) ---
    print("\n>>> 🧪 Validando Modelo Final no Conjunto de Teste (Cego)...")
    final_model = RandomForestClassifier(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        random_state=DEFAULT_SEED
    )
    final_model.fit(X_train_global, y_train_global)
    y_pred = final_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    # --- 6. Salvar Resultados ---
    
    # 6.1 Salvar CSV de histórico
    csv_path = os.path.join(REPORTS_PATH, "pso_history.csv")
    df_history.to_csv(csv_path, index=False)
    print(f"📄 Histórico salvo em: {csv_path}")

    # 6.2 Salvar Gráfico de Convergência
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_history, x='iteration', y='score', marker='o', label='Partículas')
    # Linha do melhor global por iteração
    best_per_iter = df_history.groupby('iteration')['score'].max()
    plt.plot(best_per_iter.index, best_per_iter.values, 'r--', linewidth=2, label='Melhor Global')
    plt.title('Convergência do PSO - Acurácia Random Forest')
    plt.xlabel('Iteração')
    plt.ylabel('Acurácia (CV)')
    plt.legend()
    plt.grid(True)
    img_path = os.path.join(REPORTS_PATH, "pso_convergence.png")
    plt.savefig(img_path)
    print(f"📊 Gráfico salvo em: {img_path}")

    # 6.3 Relatório Final
    report = f"""
    === RESULTADOS PSO ===
    Melhores Parâmetros: n_estimators={best_n_estimators}, max_depth={best_max_depth}
    Acurácia no Treino (CV): {best_cv_score:.4f}
    Acurácia no Teste (Cego): {test_acc:.4f}
    
    Diferença (Treino - Teste): {(best_cv_score - test_acc):.4f}
    (Se a diferença for < 0.05, não há overfitting severo)
    """
    print(report)
    with open(os.path.join(REPORTS_PATH, "pso_final_result.txt"), "w") as f:
        f.write(report)