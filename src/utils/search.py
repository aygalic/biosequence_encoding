from . import experiment
import random


class genetic_search:
    def __init__(self, search_param, data_param, model_param, initial_population, dynamic_params):
        # param in relation with the genetic algorithm search (hyper param)
        self.search_param = search_param

        # params to be plugged into the experiment and optimized
        self.data_param = data_param
        self.model_param = model_param
        self.dynamic_params = dynamic_params

        self.alt_data_param = None

        # initialization
        self.initial_population = initial_population
        self.best_performer = None
        self.best_performer_metric = 0 # we start from 0 instead of -inf
        self.current_generation = 0
        self.performance_tracker = [0]*self.search_param["generations"]
        self.n_iter = 0
        
    def add_alternative_dataset(self, alt_data_param):
        self.alt_data_param = alt_data_param


    def calculate_fitness(self, individual):
        
        e = experiment.Experiment(data_param=self.data_param, model_param=individual, n_epoch = self.search_param['EPOCH'])
        e.run()    
        if (e.metric >= self.best_performer_metric):
            print("new best performer :", individual)
            print("Score achieved :", e.metric)
            self.best_performer = individual
            self.best_performer_metric = e.metric

            if self.performance_tracker[self.current_generation] <= e.metric :
                self.performance_tracker[self.current_generation] = e.metric
            
        else:   
            print("No change in best performer.")
            print("Best individual is still:", self.best_performer, "with metric:", self.best_performer_metric)
        
        if self.alt_data_param is not None:
            e = experiment.Experiment(data_param=self.alt_data_param, model_param=individual, n_epoch = self.search_param['EPOCH'])
            e.run() 
        self.n_iter += 1
        return e.metric

    def select_parents(self, population):
        # Assuming your population is a list of individuals and 
        # you can retrieve the fitness of each individual, possibly through a function call.

        # First, sort the population by fitness. I'm assuming higher fitness is better.
        # If your fitness measure works the other way, you can reverse the sort.
        sorted_population = sorted(population, key=self.calculate_fitness, reverse=True)

        # Now, select the two fittest individuals. If you prefer, you could also add some
        # stochastic behavior in this selection (e.g., sometimes choosing individuals
        # other than the absolute fittest).
        parent1 = sorted_population[0]
        parent2 = sorted_population[1] if sorted_population[1] != parent1 else sorted_population[2]

        return parent1, parent2

    # most basic crossover possible
    def crossover(self, parent1, parent2):
        child = {}
        for key in parent1.keys():
            child[key] = parent1[key] if random.choice([True, False]) else parent2[key]
        
        return child

    def mutate(self, individual):
        # Choose a parameter to mutate
        mutation_param_key = random.choice(list(self.dynamic_params.keys()))
            
        # if we find a tuple of coupled parametter, we have to adjust our approach
        if isinstance(self.dynamic_params[mutation_param_key][0], tuple):
            # Choose a new set of values for the coupled parameters, ensuring it's not the same as the current values
            current_values = [(key, individual[key]) for key, _ in self.dynamic_params[mutation_param_key][0]]
            available_choices = [val for val in self.dynamic_params[mutation_param_key] if val != current_values]

            if not available_choices:
                return individual  # No mutation if there are no other options

            new_values = random.choice(available_choices)

            # Apply the mutation
            mutated_individual = individual.copy()
            for (key, val) in new_values:
                mutated_individual[key] = val  # Update each of the coupled parameters

        else:

            # Choose a new value for the parameter from the provided list, making sure it's not the same as the current value
            current_value = individual[mutation_param_key]
            available_choices = [val for val in self.dynamic_params[mutation_param_key] if val != current_value]

            # If there are no available choices (e.g., list had only one element), no mutation happens
            if not available_choices:
                return individual

            new_value = random.choice(available_choices)

            # Apply the mutation
            mutated_individual = individual.copy()
            mutated_individual[mutation_param_key] = new_value

        return mutated_individual
    
    def run(self):
        self.current_generation = 0
        population = self.initial_population

        for generation in range(self.search_param["generations"]):
            print(f"Running generation {generation + 1}...")


            new_population = []
            while len(new_population) < self.search_param['population_size']:
                # Selection
                parent1, parent2 = self.select_parents(population)

                # Crossover
                if random.random() < self.search_param["crossover_rate"]:
                    child = self.crossover(parent1, parent2)
                else:
                    # If no crossover, just select one of the parents at random for next generation
                    child = random.choice([parent1, parent2])

                # Mutation
                if random.random() < self.search_param["mutation_rate"]:
                    child = self.mutate(child)

                new_population.append(child)

            # Here, you may want to mix the new population with the old one and keep the best for next generation
            # Or completely replace it, depending on your strategy.

            population = new_population
            self.current_generation += 1





"""
class random_search:
    def __init__(self, search_param, configs, param_sets):
        self.search_param = search_param

class exhaustive_search:
    def __init__(self, search_param, configs, param_sets):
        self.search_param = search_param
"""