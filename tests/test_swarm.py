"""
Testes unitarios para os algoritmos da Parte 4:
- PSO (Particle Swarm Optimization)
- CLONALG (Sistema Imune Artificial)
"""

import sys
import os
import pytest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.part4_swarm_immune.pso_tunning import PSO, Particle
from src.part4_swarm_immune.immune_tuning import CLONALG

# -------------------------------------------------------------------------
# FUNCOES AUXILIARES DE TESTE
# -------------------------------------------------------------------------

def mock_fitness_sphere(params):
    """
    Funcao Esfera: minimiza sum(x^2). Otimo em [0, 0].
    Para maximizacao, retornamos o negativo.
    """
    return -sum(p**2 for p in params)

def mock_fitness_simple(params):
    """Funcao simples: maximiza a soma dos parametros."""
    return sum(params)

# -------------------------------------------------------------------------
# TESTES DO PSO
# -------------------------------------------------------------------------

class TestPSO:
    
    def test_particle_initialization(self):
        """Verifica se as particulas sao inicializadas dentro dos bounds."""
        bounds = [(0, 10), (0, 100)]
        
        for _ in range(50):
            p = Particle(bounds)
            assert 0 <= p.position[0] <= 10, "Particula fora do bound 0"
            assert 0 <= p.position[1] <= 100, "Particula fora do bound 1"
    
    def test_pso_initialization(self):
        """Verifica se o PSO inicializa corretamente."""
        bounds = [(0, 10), (0, 100)]
        pso = PSO(mock_fitness_simple, bounds, n_particles=5, max_iter=3)
        
        assert pso.n_particles == 5
        assert pso.max_iter == 3
        assert pso.global_best_score == -np.inf
    
    def test_pso_optimization_simple(self):
        """Testa se o PSO consegue otimizar um problema simples."""
        bounds = [(0, 100), (0, 100)]
        pso = PSO(mock_fitness_simple, bounds, n_particles=10, max_iter=10)
        
        best_pos, best_score, history = pso.optimize()
        
        # O otimo e [100, 100] com fitness = 200
        # Aceitamos qualquer coisa acima de 150
        assert best_score > 150, f"PSO nao convergiu bem. Score: {best_score}"
    
    def test_pso_respects_bounds(self):
        """Verifica se o PSO mantem as particulas dentro dos bounds."""
        bounds = [(0, 10), (0, 10)]
        pso = PSO(mock_fitness_simple, bounds, n_particles=10, max_iter=20)
        
        best_pos, best_score, history = pso.optimize()
        
        # Verifica se todas as posicoes no historico estao dentro dos bounds
        for _, row in history.iterrows():
            assert 0 <= row['C'] <= 10, f"C fora dos bounds: {row['C']}"
            assert 0 <= row['gamma'] <= 10, f"Gamma fora dos bounds: {row['gamma']}"
    
    def test_pso_history_structure(self):
        """Verifica se o historico tem a estrutura correta."""
        bounds = [(0, 10), (0, 10)]
        pso = PSO(mock_fitness_simple, bounds, n_particles=5, max_iter=3)
        
        _, _, history = pso.optimize()
        
        assert 'iteration' in history.columns
        assert 'C' in history.columns
        assert 'gamma' in history.columns
        assert 'score' in history.columns
        assert len(history) == 5 * 3  # n_particles * max_iter


# -------------------------------------------------------------------------
# TESTES DO CLONALG
# -------------------------------------------------------------------------

class TestCLONALG:
    
    def test_clonalg_initialization(self):
        """Verifica se o CLONALG inicializa corretamente."""
        bounds = [(0, 10), (0, 100)]
        clonalg = CLONALG(mock_fitness_simple, bounds, pop_size=10, max_iter=5)
        
        assert len(clonalg.population) == 10
        assert clonalg.max_iter == 5
        assert clonalg.best_score == -np.inf
    
    def test_clonalg_population_bounds(self):
        """Verifica se a populacao inicial respeita os bounds."""
        bounds = [(0, 10), (100, 200)]
        clonalg = CLONALG(mock_fitness_simple, bounds, pop_size=20, max_iter=1)
        
        for ab in clonalg.population:
            assert 0 <= ab[0] <= 10, f"Gene 0 fora do bound: {ab[0]}"
            assert 100 <= ab[1] <= 200, f"Gene 1 fora do bound: {ab[1]}"
    
    def test_clonalg_mutation(self):
        """Verifica se a mutacao respeita os bounds."""
        bounds = [(0, 10), (0, 10)]
        clonalg = CLONALG(mock_fitness_simple, bounds, pop_size=5, max_iter=1)
        
        antibody = np.array([5.0, 5.0])
        
        # Testa 100 mutacoes
        for _ in range(100):
            mutated = clonalg.mutate(antibody.copy(), rank=0, total_ranks=5)
            assert 0 <= mutated[0] <= 10, f"Mutacao violou bound 0: {mutated[0]}"
            assert 0 <= mutated[1] <= 10, f"Mutacao violou bound 1: {mutated[1]}"
    
    def test_clonalg_optimization_simple(self):
        """Testa se o CLONALG consegue otimizar um problema simples."""
        bounds = [(0, 100), (0, 100)]
        clonalg = CLONALG(mock_fitness_simple, bounds, pop_size=10, 
                          selection_size=5, clones_per_factor=2, max_iter=10)
        
        best_pos, best_score, history = clonalg.optimize()
        
        # O otimo e [100, 100] com fitness = 200
        # Aceitamos qualquer coisa acima de 150
        assert best_score > 150, f"CLONALG nao convergiu bem. Score: {best_score}"
    
    def test_clonalg_history_structure(self):
        """Verifica se o historico tem a estrutura correta."""
        bounds = [(0, 10), (0, 10)]
        clonalg = CLONALG(mock_fitness_simple, bounds, pop_size=5, max_iter=3)
        
        _, _, history = clonalg.optimize()
        
        assert 'generation' in history.columns
        assert 'C' in history.columns
        assert 'gamma' in history.columns
        assert 'score' in history.columns
    
    def test_clonalg_elitism(self):
        """Verifica se o melhor anticorpo e preservado (elitismo)."""
        bounds = [(0, 100), (0, 100)]
        clonalg = CLONALG(mock_fitness_simple, bounds, pop_size=10, 
                          selection_size=3, max_iter=5)
        
        best_pos, best_score, history = clonalg.optimize()
        
        # O best_score final deve ser >= ao melhor de cada geracao
        best_per_gen = history.groupby('generation')['score'].max()
        for gen_best in best_per_gen:
            assert best_score >= gen_best, "Elitismo falhou - perdeu o melhor anticorpo"


# -------------------------------------------------------------------------
# TESTES DE INTEGRACAO (ARQUIVOS GERADOS)
# -------------------------------------------------------------------------

class TestPart4Integration:
    
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    REPORTS_DIR = os.path.join(BASE_DIR, "reports", "part4_swarm_immune")
    FIGS_DIR = os.path.join(BASE_DIR, "reports", "figs")
    
    def test_pso_history_file_exists(self):
        """Verifica se o arquivo de historico do PSO foi gerado."""
        path = os.path.join(self.REPORTS_DIR, "pso_history.csv")
        assert os.path.exists(path), "pso_history.csv nao encontrado. Rode 'make part4'."
    
    def test_immune_history_file_exists(self):
        """Verifica se o arquivo de historico do CLONALG foi gerado."""
        path = os.path.join(self.REPORTS_DIR, "immune_history.csv")
        assert os.path.exists(path), "immune_history.csv nao encontrado. Rode 'make part4'."
    
    def test_pso_convergence_plot_exists(self):
        """Verifica se o grafico de convergencia do PSO foi gerado."""
        path = os.path.join(self.FIGS_DIR, "pso_convergence.png")
        assert os.path.exists(path), "pso_convergence.png nao encontrado. Rode 'make part4'."
    
    def test_immune_convergence_plot_exists(self):
        """Verifica se o grafico de convergencia do CLONALG foi gerado."""
        path = os.path.join(self.FIGS_DIR, "immune_convergence.png")
        assert os.path.exists(path), "immune_convergence.png nao encontrado. Rode 'make part4'."
