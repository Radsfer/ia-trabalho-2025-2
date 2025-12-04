import random
import numpy as np
import copy

class GeneticAlgorithm:
    def __init__(self, pop_size, mutation_rate, crossover_rate, elitism_count, fitness_func, gene_bounds):
        """
        :param pop_size: Tamanho da população (ex: 10 ou 20)
        :param fitness_func: Função que recebe um indivíduo e retorna uma nota (float)
        :param gene_bounds: Lista de tuplas com min/max para cada gene. Ex: [(0.1, 100), (0.001, 1)]
        """
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.fitness_func = fitness_func
        self.bounds = gene_bounds
        
        # Inicializa população aleatória dentro dos limites
        self.population = []
        for _ in range(pop_size):
            ind = []
            for min_val, max_val in self.bounds:
                ind.append(random.uniform(min_val, max_val))
            self.population.append(ind)
            
        self.best_solution = None
        self.best_fitness = -float('inf')
        self.history = []

    def evaluate(self):
        """Calcula o fitness de toda a população e ordena."""
        pop_with_fitness = []
        for ind in self.population:
            fit = self.fitness_func(ind)
            pop_with_fitness.append((ind, fit))
            
            # Atualiza o melhor global
            if fit > self.best_fitness:
                self.best_fitness = fit
                self.best_solution = ind
        
        # Ordena do maior fitness para o menor
        pop_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Guarda apenas os indivíduos ordenados na população
        self.population = [p[0] for p in pop_with_fitness]
        self.history.append(self.best_fitness)
        
        return self.best_fitness

    def select(self):
        """Torneio: Pega 3 aleatórios e escolhe o melhor."""
        candidates = random.sample(self.population, 3)
        # Como já avaliamos antes, poderíamos otimizar, mas recalculando simplifica o código didático
        # Para ser eficiente, aqui assumimos que self.population já está ordenado pelo evaluate()
        # Mas para o torneio ser justo, pegamos a fitness function de novo ou confiamos na sorte.
        # Vamos fazer um torneio simples comparando fitness recalculado (ou cacheado se fosse complexo).
        
        best = candidates[0]
        best_f = self.fitness_func(best)
        
        for c in candidates[1:]:
            f = self.fitness_func(c)
            if f > best_f:
                best = c
                best_f = f
        return best

    def crossover(self, p1, p2):
        """Crossover Aritmético (Média) ou Ponto Único."""
        if random.random() < self.crossover_rate:
            # Vamos fazer um crossover simples: média ponderada
            alpha = random.random()
            child = []
            for i in range(len(p1)):
                val = alpha * p1[i] + (1 - alpha) * p2[i]
                child.append(val)
            return child
        else:
            return p1[:] # Retorna cópia do pai 1 se não cruzar

    def mutate(self, ind):
        """Mutação Gaussiana: adiciona um pequeno ruído."""
        for i in range(len(ind)):
            if random.random() < self.mutation_rate:
                # Ruído baseado na escala do gene
                r_range = self.bounds[i][1] - self.bounds[i][0]
                noise = random.gauss(0, r_range * 0.1) # 10% da escala
                ind[i] += noise
                
                # Clamp (Garante que não foge dos limites)
                ind[i] = max(self.bounds[i][0], min(ind[i], self.bounds[i][1]))
        return ind

    def run(self, generations):
        print(f"--- Iniciando AG ({generations} gerações) ---")
        for g in range(generations):
            current_best = self.evaluate()
            print(f"Gen {g+1}: Melhor Fitness = {current_best:.4f} | Genes: {[round(x,4) for x in self.best_solution]}")
            
            new_pop = []
            
            # Elitismo: Mantém os melhores intocados
            for i in range(self.elitism_count):
                new_pop.append(self.population[i])
            
            # Gera o resto da população
            while len(new_pop) < self.pop_size:
                p1 = self.select()
                p2 = self.select()
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_pop.append(child)
            
            self.population = new_pop
            
        return self.best_solution, self.best_fitness