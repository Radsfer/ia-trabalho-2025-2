import random
import copy

class GeneticAlgorithm:
    def __init__(self, pop_size, mutation_rate, crossover_rate, elitism_count, fitness_func, gene_bounds):
        """
        :param pop_size: Tamanho da população
        :param fitness_func: Função que recebe indivíduo e retorna fitness (float)
        :param gene_bounds: Lista de tuplas com min/max para cada gene
        """
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.fitness_func = fitness_func
        self.bounds = gene_bounds
        
        # Inicializa população aleatória
        self.population = []
        for _ in range(pop_size):
            ind = []
            for min_val, max_val in self.bounds:
                ind.append(random.uniform(min_val, max_val))
            self.population.append(ind)
            
        self.best_solution = None
        self.best_fitness = -float('inf')
        self.history = [] # Aqui guardamos o melhor fitness de cada geração

    def evaluate(self):
        """Calcula fitness e ordena a população."""
        pop_with_fitness = []
        for ind in self.population:
            fit = self.fitness_func(ind)
            pop_with_fitness.append((ind, fit))
            
            if fit > self.best_fitness:
                self.best_fitness = fit
                self.best_solution = ind
        
        # Ordena do maior para o menor
        pop_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Atualiza população ordenada e histórico
        self.population = [p[0] for p in pop_with_fitness]
        
        # Guarda o melhor desta geração no histórico
        current_best_val = pop_with_fitness[0][1]
        self.history.append(current_best_val)
        
        return current_best_val

    def select(self):
        """Torneio simples."""
        candidates = random.sample(self.population, 3)
        # Vamos reavaliar para ser seguro (custo baixo aqui)
        best = candidates[0]
        best_f = self.fitness_func(best)
        for c in candidates[1:]:
            f = self.fitness_func(c)
            if f > best_f:
                best = c
                best_f = f
        return best

    def crossover(self, p1, p2):
        if random.random() < self.crossover_rate:
            alpha = random.random()
            child = []
            for i in range(len(p1)):
                val = alpha * p1[i] + (1 - alpha) * p2[i]
                child.append(val)
            return child
        return p1[:]

    def mutate(self, ind):
        for i in range(len(ind)):
            if random.random() < self.mutation_rate:
                r_range = self.bounds[i][1] - self.bounds[i][0]
                noise = random.gauss(0, r_range * 0.1)
                ind[i] += noise
                ind[i] = max(self.bounds[i][0], min(ind[i], self.bounds[i][1]))
        return ind

    def run(self, generations):
        print(f"--- Iniciando AG ({generations} gerações) ---")
        for g in range(generations):
            current_best = self.evaluate()
            print(f"Gen {g+1}: Melhor Fitness = {current_best:.4f}")
            
            new_pop = []
            # Elitismo
            for i in range(self.elitism_count):
                new_pop.append(self.population[i])
            
            # Gera o resto
            while len(new_pop) < self.pop_size:
                p1 = self.select()
                p2 = self.select()
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_pop.append(child)
            
            self.population = new_pop
            
        # AGORA RETORNA O HISTÓRICO TAMBÉM
        return self.best_solution, self.best_fitness, self.history