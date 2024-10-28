The provided Python code implements a **Genetic Algorithm (GA)** to optimize the Rastrigin function, a well-known test function in optimization problems. Here’s a detailed breakdown of the code, explaining its components, functionality, and how it operates.

### Libraries Used
```python
import numpy as np
import random
```
- **NumPy** (`import numpy as np`): A library used for numerical operations, particularly for array manipulation and mathematical functions.
- **Random** (`import random`): Although imported, it is not used in the code. NumPy's random functions suffice for this implementation.

### Rastrigin Function Definition
```python
def rastrigin_function(X):
    A = 10
    return A * len(X) + sum([(x ** 2 - A * np.cos(2 * np.pi * x)) for x in X])
```
- The **Rastrigin function** is defined, which is commonly used for benchmarking optimization algorithms.
- The function formula is given by:
  \[
  f(X) = An + \sum_{i=1}^{n}(x_i^2 - A \cos(2 \pi x_i))
  \]
  where \(A = 10\) and \(n\) is the number of variables (length of \(X\)).
- The function is multimodal and has a global minimum at \(X = 0\), where \(f(X) = 0\).

### Genetic Algorithm Class
```python
class GeneticAlgorithm:
    def __init__(self, fitness_func, population_size, gene_length, crossover_rate, mutation_rate, generations):
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.gene_length = gene_length  # Number of genes (variables in optimization)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = self.initialize_population()
```
- The **GeneticAlgorithm** class encapsulates all the necessary functionalities to run the GA.
- **Parameters**:
  - `fitness_func`: The function used to evaluate the fitness of individuals.
  - `population_size`: The number of individuals in each generation.
  - `gene_length`: The number of variables (genes) each individual has.
  - `crossover_rate`: The probability of crossover between parent individuals.
  - `mutation_rate`: The probability of mutation for each gene.
  - `generations`: The number of generations to evolve the population.

### Population Initialization
```python
def initialize_population(self):
    # Initialize population with random values within a specified range
    return np.random.uniform(-5.12, 5.12, (self.population_size, self.gene_length))
```
- The `initialize_population` method creates an initial population of individuals (solutions) with random values uniformly distributed between \([-5.12, 5.12]\), the bounds of the Rastrigin function.

### Evaluate Population
```python
def evaluate_population(self):
    # Evaluate fitness for each individual
    return np.array([self.fitness_func(individual) for individual in self.population])
```
- The `evaluate_population` method computes the fitness of each individual by applying the `fitness_func` (Rastrigin function) to every member of the population.

### Parent Selection
```python
def select_parents(self):
    # Use tournament selection for choosing parents
    selected = []
    for _ in range(self.population_size):
        i, j = np.random.choice(range(self.population_size), size=2, replace=False)
        if self.fitness[i] < self.fitness[j]:  # Minimization problem
            selected.append(self.population[i])
        else:
            selected.append(self.population[j])
    return np.array(selected)
```
- The `select_parents` method employs **tournament selection** to choose parents for the next generation:
  - Two individuals are randomly selected, and the one with the better fitness (lower Rastrigin value) is chosen as a parent.
  - This process repeats until a full parent set is selected.

### Crossover Function
```python
def crossover(self, parent1, parent2):
    if np.random.rand() < self.crossover_rate:
        # Single-point crossover
        point = np.random.randint(1, self.gene_length - 1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    else:
        return parent1, parent2
```
- The `crossover` method performs a **single-point crossover**:
  - A random crossover point is selected, and genetic material (genes) from both parents is combined to create two offspring (children).
  - If the random number is greater than the `crossover_rate`, parents are returned unchanged.

### Mutation Function
```python
def mutate(self, individual):
    # Mutation by adding Gaussian noise
    for i in range(self.gene_length):
        if np.random.rand() < self.mutation_rate:
            individual[i] += np.random.normal(0, 0.1)  # Small mutation step
            # Keep individual within bounds
            individual[i] = np.clip(individual[i], -5.12, 5.12)
    return individual
```
- The `mutate` method applies mutations to an individual by adding Gaussian noise:
  - Each gene is mutated with a probability defined by the `mutation_rate`.
  - The mutation is a small random adjustment, and the values are clipped to ensure they remain within the bounds of the Rastrigin function.

### Running the Genetic Algorithm
```python
def run(self):
    for generation in range(self.generations):
        # Evaluate fitness of population
        self.fitness = self.evaluate_population()

        # Elitism: Keep the best individual
        best_idx = np.argmin(self.fitness)
        best_individual = self.population[best_idx].copy()
        best_fitness = self.fitness[best_idx]

        # Select parents
        parents = self.select_parents()

        # Generate next generation with crossover and mutation
        next_generation = []
        for i in range(0, self.population_size, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = self.crossover(parent1, parent2)
            next_generation.append(self.mutate(child1))
            next_generation.append(self.mutate(child2))

        self.population = np.array(next_generation[:self.population_size])

        # Reinsert the best individual (elitism)
        self.population[0] = best_individual

        # Display progress
        print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

    # Final best solution
    self.fitness = self.evaluate_population()
    best_idx = np.argmin(self.fitness)
    best_solution = self.population[best_idx]
    best_fitness = self.fitness[best_idx]
    return best_solution, best_fitness
```
- The `run` method orchestrates the entire GA process:
  - For each generation:
    - The fitness of the current population is evaluated.
    - The best individual (with the lowest fitness) is saved for elitism, ensuring it survives to the next generation.
    - Parents are selected for breeding.
    - The next generation is generated through crossover and mutation.
    - The best individual is reinserted into the population.
    - Progress is printed for each generation.
- At the end of the run, the best solution and its fitness value are returned.

### Parameters and Execution
```python
# Parameters
population_size = 50
gene_length = 5  # Number of variables in the Rastrigin function
crossover_rate = 0.8
mutation_rate = 0.05
generations = 100

# Run Genetic Algorithm
ga = GeneticAlgorithm(fitness_func=rastrigin_function, population_size=population_size,
                      gene_length=gene_length, crossover_rate=crossover_rate,
                      mutation_rate=mutation_rate, generations=generations)

best_solution, best_fitness = ga.run()
print("\nBest Solution:", best_solution)
print("Best Fitness (Minimum Rastrigin Value):", best_fitness)
```
- Parameters are defined to set up the genetic algorithm:
  - `population_size`: 50 individuals in each generation.
  - `gene_length`: 5 genes, corresponding to 5 variables in the Rastrigin function.
  - `crossover_rate`: 0.8, indicating an 80% chance of crossover occurring.
  - `mutation_rate`: 0.05, indicating a 5% chance of mutation for each gene.
  - `generations`: The algorithm will run for 100 generations.

- An instance of the `GeneticAlgorithm` class is created, and the `run` method is invoked.
- The final best solution and its fitness value are printed.

### Summary
- This code provides a comprehensive implementation of a Genetic Algorithm for optimizing the Rastrigin function.
- It includes all major components of a GA: population initialization, fitness evaluation, parent selection, crossover, mutation, and elitism.
- By tuning the parameters, users can experiment with different settings and observe how the algorithm converges to the minimum of the Rastrigin function.
- The overall structure is modular, making it easy to modify or extend for other optimization problems or genetic algorithm variations.
