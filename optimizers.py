import numpy as np
from scipy.stats import levy
import config
from array_utils import ArrayEvaluator

class BaseOptimizer:
    """Optimizer base class"""
    def __init__(self, fitness_func, bounds):
        self.fitness_func = fitness_func
        self.bounds = bounds
        self.dim = config.MIC_COUNT * 2
        self.pop_size = config.OPTIMIZER_POP_SIZE
        self.max_iter = config.OPTIMIZER_EPOCHS
        self.best_solution = None
        self.best_fitness = -np.inf
        
        # Initialize population
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        self.enforce_circle_constraint()

    def enforce_circle_constraint(self):
        """Ensure all microphones are within circular aperture"""
        radius = config.ARRAY_DIAMETER / 2
        for i in range(self.pop_size):
            pos_2d = self.population[i].reshape(config.MIC_COUNT, 2)
            distances = np.linalg.norm(pos_2d, axis=1)
            mask = distances > radius
            if np.any(mask):
                # Pull points outside boundary back to circumference
                pos_2d[mask] = pos_2d[mask] * (radius / distances[mask, np.newaxis])
            self.population[i] = pos_2d.flatten()

    def run(self):
        raise NotImplementedError

    def _evaluate_population(self):
        fitness_values = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            positions = self.population[i].reshape(config.MIC_COUNT, 2)
            fitness_values[i] = self.fitness_func(positions)
        return fitness_values

class GeneticOptimizer(BaseOptimizer):
    """Genetic Algorithm (GA)"""
    def run(self):
        for it in range(self.max_iter):
            fitness = self._evaluate_population()
            
            if np.max(fitness) > self.best_fitness:
                self.best_fitness = np.max(fitness)
                self.best_solution = self.population[np.argmax(fitness)].copy()

            # Normalize fitness values for roulette wheel selection
            fitness_norm = fitness - np.min(fitness)
            if np.sum(fitness_norm) > 0:
                probs = fitness_norm / np.sum(fitness_norm)
            else:
                probs = np.ones(self.pop_size) / self.pop_size
            
            indices = np.random.choice(self.pop_size, size=self.pop_size, p=probs)
            new_population = self.population[indices]

            for i in range(0, self.pop_size, 2):
                if np.random.rand() < config.GA_CROSSOVER_PROB:
                    p1, p2 = new_population[i], new_population[i+1]
                    crossover_point = np.random.randint(1, self.dim)
                    new_population[i, crossover_point:] = p2[crossover_point:]
                    new_population[i+1, crossover_point:] = p1[crossover_point:]

            mutation_mask = np.random.rand(self.pop_size, self.dim) < config.GA_MUTATION_PROB
            mutation_values = np.random.normal(0, config.GA_MUTATION_STRENGTH, (self.pop_size, self.dim))
            new_population[mutation_mask] += mutation_values[mutation_mask]
            
            self.population = np.clip(new_population, self.bounds[0], self.bounds[1])
            self.enforce_circle_constraint()

        return self.best_solution.reshape(config.MIC_COUNT, 2)

class PSOptimizer(BaseOptimizer):
    """Particle Swarm Optimization (PSO)"""
    def run(self):
        velocities = np.zeros((self.pop_size, self.dim))
        personal_best_pos = self.population.copy()
        personal_best_fit = self._evaluate_population()
        
        self.best_fitness = np.max(personal_best_fit)
        self.best_solution = personal_best_pos[np.argmax(personal_best_fit)].copy()

        for it in range(self.max_iter):
            # Linearly decreasing inertia weight
            inertia = config.PSO_INERTIA_START - (config.PSO_INERTIA_START - config.PSO_INERTIA_END) * (it / self.max_iter)
            
            r1, r2 = np.random.rand(2)
            cognitive_vel = config.PSO_C1 * r1 * (personal_best_pos - self.population)
            social_vel = config.PSO_C2 * r2 * (self.best_solution - self.population)
            velocities = inertia * velocities + cognitive_vel + social_vel
            
            self.population += velocities
            self.population = np.clip(self.population, self.bounds[0], self.bounds[1])
            self.enforce_circle_constraint()
            
            fitness = self._evaluate_population()
            
            mask = fitness > personal_best_fit
            personal_best_pos[mask] = self.population[mask]
            personal_best_fit[mask] = fitness[mask]

            if np.max(personal_best_fit) > self.best_fitness:
                self.best_fitness = np.max(personal_best_fit)
                self.best_solution = personal_best_pos[np.argmax(personal_best_fit)].copy()

        return self.best_solution.reshape(config.MIC_COUNT, 2)

class HippopotamusOptimizer(BaseOptimizer):
    """Hippopotamus Optimization Algorithm (HO)"""
    def run(self):
        fitness = self._evaluate_population()
        self.best_fitness = np.max(fitness)
        self.best_solution = self.population[np.argmax(fitness)].copy()

        for t in range(self.max_iter):
            # Phase 1: River and Pond (Exploration)
            for i in range(self.pop_size // 2):
                y1, I1 = np.random.rand(), np.random.randint(1, 3)
                new_pos = self.population[i] + y1 * (self.best_solution - I1 * self.population[i])
                
                new_pos = np.clip(new_pos, self.bounds[0], self.bounds[1])
                new_fit = self.fitness_func(new_pos.reshape(config.MIC_COUNT, 2))
                if new_fit > fitness[i]:
                    self.population[i], fitness[i] = new_pos, new_fit

            # Phase 2: Defense against Predators (Balance)
            for i in range(self.pop_size // 2, self.pop_size):
                predator_pos = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
                RL = levy.rvs(size=self.dim) * 0.01
                
                fc, g = 2 * np.random.rand() - 1, 2 - t * (2 / self.max_iter)
                d, Fi = np.random.rand(), np.random.rand()

                if Fi < 0.5:
                    new_pos = predator_pos + fc * (d * np.cos(2 * np.pi * g) - self.population[i])
                else:
                    new_pos = predator_pos + fc * (d * np.cos(2 * np.pi * g) - RL)
                
                new_pos = np.clip(new_pos, self.bounds[0], self.bounds[1])
                new_fit = self.fitness_func(new_pos.reshape(config.MIC_COUNT, 2))
                if new_fit > fitness[i]:
                    self.population[i], fitness[i] = new_pos, new_fit
            
            # Phase 3: Escape from Predators (Exploitation)
            for i in range(self.pop_size):
                y2, I2 = np.random.rand(), np.random.randint(1, 3)
                new_pos = self.best_solution + y2 * (self.population[i] - I2 * self.best_solution)

                new_pos = np.clip(new_pos, self.bounds[0], self.bounds[1])
                new_fit = self.fitness_func(new_pos.reshape(config.MIC_COUNT, 2))
                if new_fit > fitness[i]:
                    self.population[i], fitness[i] = new_pos, new_fit

            self.enforce_circle_constraint()
            
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > self.best_fitness:
                self.best_fitness = fitness[current_best_idx]
                self.best_solution = self.population[current_best_idx].copy()

        return self.best_solution.reshape(config.MIC_COUNT, 2)

def run_all_optimizers():
    """Run all defined optimizers"""
    radius = config.ARRAY_DIAMETER / 2
    bounds = [-radius, radius]
    
    # Static method to get fitness function
    fitness_func = ArrayEvaluator.get_fitness_function()
    
    optimized_geometries = {}
    
    optimizers_to_run = {
        "GA Optimized Array": GeneticOptimizer(fitness_func, bounds),
        "PSO Optimized Array": PSOptimizer(fitness_func, bounds),
        "HO Optimized Array": HippopotamusOptimizer(fitness_func, bounds)
    }

    for name, optimizer in optimizers_to_run.items():
        optimized_pos = optimizer.run()
        optimized_geometries[name] = optimized_pos
        
    return optimized_geometries
