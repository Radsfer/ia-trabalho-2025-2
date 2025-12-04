import sys
import os
import pytest
import random
from src.part3_ga.ga import GeneticAlgorithm

# Adiciona o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# -------------------------------------------------------------------------
# MOCKS E FUNÇÕES AUXILIARES
# -------------------------------------------------------------------------

def mock_fitness_max(genes):
    """
    Função de fitness simples: O objetivo é maximizar o valor do gene.
    Ex: Gene [10.0] é melhor que [5.0].
    """
    return sum(genes)

# -------------------------------------------------------------------------
# TESTES DE LÓGICA DO AG
# -------------------------------------------------------------------------

def test_initialization():
    """Prova que a população inicial respeita os limites (bounds)."""
    bounds = [(0, 10), (100, 200)] # Gene 1 entre 0-10, Gene 2 entre 100-200
    pop_size = 20
    
    ga = GeneticAlgorithm(
        pop_size=pop_size,
        mutation_rate=0.1,
        crossover_rate=0.5,
        elitism_count=2,
        fitness_func=mock_fitness_max,
        gene_bounds=bounds
    )
    
    assert len(ga.population) == pop_size
    
    for ind in ga.population:
        # Verifica Gene 1
        assert 0 <= ind[0] <= 10
        # Verifica Gene 2
        assert 100 <= ind[1] <= 200

def test_elitism_preservation():
    """Prova que o melhor indivíduo NÃO é perdido na próxima geração."""
    ga = GeneticAlgorithm(
        pop_size=10,
        mutation_rate=0.5,
        crossover_rate=0.5,
        elitism_count=1, # Mantém o top 1
        fitness_func=mock_fitness_max,
        gene_bounds=[(0, 100)]
    )
    
    # Cria uma população falsa onde sabemos quem é o melhor
    ga.population = [[10.0], [5.0], [1.0], [90.0]] + [[0.0]]*6
    # O [90.0] é o campeão
    
    # Roda a avaliação (ordena)
    ga.evaluate()
    best_fitness_gen1 = ga.best_fitness
    assert best_fitness_gen1 == 90.0
    
    # Avança uma geração (sem rodar o loop completo, simulando passos)
    # Mas como o método 'run' faz tudo, vamos chamar 'run(1)'
    ga.run(generations=1)
    
    # O melhor da nova geração deve ser PELO MENOS igual ao anterior (Elitismo)
    assert ga.best_fitness >= 90.0 

def test_mutation_bounds():
    """Prova que a mutação não viola os limites (ex: Gamma negativo)."""
    # Limites estritos: 0 a 10
    ga = GeneticAlgorithm(
        pop_size=1,
        mutation_rate=1.0, # 100% de chance de mutar
        crossover_rate=0.0,
        elitism_count=0,
        fitness_func=mock_fitness_max,
        gene_bounds=[(0, 10)]
    )
    
    # Indivíduo no limite máximo
    ind = [10.0]
    
    # Tenta mutar 100 vezes
    for _ in range(100):
        mutated = ga.mutate(ind[:]) # Passa cópia
        assert 0 <= mutated[0] <= 10.0, f"Mutação violou o limite! Valor: {mutated[0]}"

def test_optimization_capability():
    """
    Prova que o AG é inteligente: Ele consegue resolver um problema fácil?
    Problema: Achar x que maximiza f(x) = x, com limite 100.
    Resposta esperada: Perto de 100.
    """
    ga = GeneticAlgorithm(
        pop_size=20,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elitism_count=2,
        fitness_func=mock_fitness_max,
        gene_bounds=[(0, 100)]
    )
    
    best_ind, best_fit = ga.run(generations=20)
    
    # Aceitamos qualquer coisa acima de 95 (o ótimo é 100)
    assert best_fit > 95, f"O AG falhou em otimizar um problema simples. Deu {best_fit}"