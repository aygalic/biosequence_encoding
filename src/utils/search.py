"""
    Implements a genetic algorithm for optimizing parameters in machine learning models.

    This class manages a population of parameter sets (individuals), evolving them over generations 
    to optimize a fitness metric evaluated through experiments.

    Attributes:
        search_param (dict): Parameters related to the genetic algorithm's operation.
        data_param (dict): Parameters for the dataset used in the experiment.
        model_param (dict): Parameters for the model used in the experiment.
        dynamic_params (dict): Parameters that are subject to mutation.
        initial_population (list): The initial set of individuals for the genetic algorithm.
        alt_data_param (dict, optional): Alternative dataset parameters for additional testing.
        best_performer (dict): The best performing individual so far.
        best_performer_metric (float): The metric score of the best performer.
        current_generation (int): The current generation count in the algorithm.
        performance_tracker (list): Tracks the best performance in each generation.
        n_iter (int): Counter for the number of iterations.

    Methods:
        add_alternative_dataset: Adds alternative dataset parameters for testing.
        calculate_fitness: Evaluates and returns the fitness of an individual.
        select_parents: Selects two parents from the population based on fitness.
        crossover: Creates a new individual by combining attributes of two parents.
        mutate: Applies mutation to an individual's parameters.
        run: Executes the genetic algorithm for the specified number of generations.
"""

from . import experiment
import random
import torch
import gc


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
        """
        Adds alternative dataset parameters for additional testing during the fitness evaluation.

        Args:
            alt_data_param (dict): Parameters for the alternative dataset.
        """
        self.alt_data_param = alt_data_param


    def calculate_fitness(self, individual):
        """
        Evaluates and returns the fitness of an individual based on an experiment.

        Fitness is determined by running an experiment with the individual's parameters and 
        measuring its performance.

        Args:
            individual (dict): A set of parameters representing an individual in the population.

        Returns:
            float: The fitness metric of the individual.
        """
        e = experiment.Experiment(data_param=self.data_param, model_param=individual)
        e.run()
        metric = e.metric

        # trying to spare my poor gpu
        del e.model
        del e 
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()

        if (metric >= self.best_performer_metric):
            print("new best performer :", individual)
            print("Score achieved :", metric)
            self.best_performer = individual
            self.best_performer_metric = metric

            if self.performance_tracker[self.current_generation] <= metric :
                self.performance_tracker[self.current_generation] = metric
            
        else:   
            print("No change in best performer.")
            print("Best individual is still:", self.best_performer, "with metric:", self.best_performer_metric)
        
        if self.alt_data_param is not None:
            for param in self.alt_data_param:
                e_alt = experiment.Experiment(data_param=param, model_param=individual)
                e_alt.run() 
                del e_alt.model
                del e_alt
                with torch.no_grad():
                    torch.cuda.empty_cache()
                gc.collect()

        self.n_iter += 1
        return metric

    def select_parents(self, population):
        """
        Selects two parents from the population based on their fitness.

        Args:
            population (list): The current population of individuals.

        Returns:
            tuple: A tuple of two individuals selected as parents.
        """

        # First, sort the population by fitness. Assuming higher fitness is better.
        sorted_population = sorted(population, key=self.calculate_fitness, reverse=True)

        # Now, select the two fittest individuals. If you prefer, you could also add some
        # stochastic behavior in this selection (e.g., sometimes choosing individuals
        # other than the absolute fittest).
        parent1 = sorted_population[0]
        parent2 = sorted_population[1] if sorted_population[1] != parent1 else sorted_population[2]

        return parent1, parent2

    # most basic crossover possible
    def crossover(self, parent1, parent2):
        """
        Creates a new individual by combining attributes of two parents.

        Args:
            parent1 (dict): The first parent's parameters.
            parent2 (dict): The second parent's parameters.

        Returns:
            dict: A new individual created from the parents.
        """
        child = {}
        for key in parent1.keys():
            child[key] = parent1[key] if random.choice([True, False]) else parent2[key]
        
        return child

    def mutate(self, individual):
        """
        Applies mutation to an individual's parameters.

        Selects a parameter at random and changes its value, based on available choices in dynamic_params.

        Args:
            individual (dict): The individual to be mutated.

        Returns:
            dict: The mutated individual.
        """
        # Choose a parameter to mutate
        mutation_param_key = random.choice(list(self.dynamic_params.keys()))
            
        # if we find a tuple of coupled parametter, we have to adjust our approach
        if isinstance(self.dynamic_params[mutation_param_key][0], tuple):
            # Choose a new set of values for the coupled parameters, ensuring it's not the same as the current values
            print("self.dynamic_params[mutation_param_key][0]", self.dynamic_params[mutation_param_key][0])
            
            available_choices = [val for val in self.dynamic_params[mutation_param_key]]
            print("available_choices", available_choices)

            if not available_choices:
                return individual  # No mutation if there are no other options

            new_values = random.choice(available_choices)
            print("new_value:",  new_values)
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
        """
        Executes the genetic algorithm, evolving the population over specified generations.

        In each generation, selects parents, performs crossover and mutation, 
        and updates the population for the next generation.
        """
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




