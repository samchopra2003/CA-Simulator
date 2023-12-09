import os

from dotenv import load_dotenv

load_dotenv()

WORLD_SIZE_ROWS = int(os.getenv("WORLD_SIZE_ROWS"))
WORLD_SIZE_COLS = int(os.getenv("WORLD_SIZE_COLS"))
STARTING_POPULATION = int(os.getenv("STARTING_POPULATION"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS"))
STEPS_PER_GEN = int(os.getenv("STEPS_PER_GEN"))
GENOME_START_LENGTH = int(os.getenv("GENOME_START_LENGTH"))
HIDDEN_NEURONS = int(os.getenv("HIDDEN_NEURONS"))
MUTATION_RATE_STRUCT = float(os.getenv("MUTATION_RATE_STRUCT"))
MUTATION_RATE_NON_STRUCT = float(os.getenv("MUTATION_RATE_NON_STRUCT"))
TIME_TO_LIVE = int(os.getenv("TIME_TO_LIVE"))


class Monitor:
    """
    Singleton Monitor Class
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Monitor, cls).__new__(cls)
            cls._instance._init_class_vars()
        return cls._instance

    def _init_class_vars(self):
        self.total_population = 0
        self.avg_fitness = 0
        self.total_mutations = 0
        self.num_predators = 0  # predator: at least one kill
        self.all_time_population = 0
        self.num_killed = 0
        self.num_reproductions = 0
        self.num_males = 0
        self.num_females = 0

        self.generation = 0
        self.gen_step = 0

        self.num_species = 0
        self.species_names = []
        self.species_populations = []
        self.species_fitnesses = []

        # fitness logger
        fitness_directory_path = 'logs/fitness'
        base_file_name = 'fitness'
        file_extension = '.txt'

        existing_files = [f for f in os.listdir(fitness_directory_path) if f.endswith(file_extension)]
        if existing_files:
            existing_files.sort()
            last_file_number = int(existing_files[-1].replace(base_file_name, '').replace(file_extension, ''))
            new_file_name = f"{base_file_name}{last_file_number + 1}{file_extension}"
        else:
            new_file_name = f"{base_file_name}0{file_extension}"

        self.fitness_file = os.path.join(fitness_directory_path, new_file_name)
        with open(self.fitness_file, 'w') as file:
            file.write(f'Params:\n'
                       '----------------\n'
                       f'World size: ({WORLD_SIZE_ROWS}, {WORLD_SIZE_COLS})\n'
                       f'Starting Population: {STARTING_POPULATION}\n'
                       f'Number of Generations: {NUM_GENERATIONS}\n'
                       f'Steps per Generation: {STEPS_PER_GEN}\n'
                       f'Genome start length: {GENOME_START_LENGTH}\n'
                       f'Number of hidden neurons: {HIDDEN_NEURONS}\n'
                       f'Mutation Rate Structural: {MUTATION_RATE_STRUCT}, Non-Structural: {MUTATION_RATE_NON_STRUCT}\n'
                       f'Time to live: {TIME_TO_LIVE}\n'
                       '----------------\n')

        self.log_fitness()

        # population logger
        population_directory_path = 'logs/population'
        base_file_name = 'population'
        file_extension = '.txt'

        existing_files = [f for f in os.listdir(population_directory_path) if f.endswith(file_extension)]
        if existing_files:
            existing_files.sort()
            last_file_number = int(existing_files[-1].replace(base_file_name, '').replace(file_extension, ''))
            new_file_name = f"{base_file_name}{last_file_number + 1}{file_extension}"
        else:
            new_file_name = f"{base_file_name}0{file_extension}"

        self.population_file = os.path.join(population_directory_path, new_file_name)
        with open(self.population_file, 'w') as file:
            file.write(f'Params:\n'
                       '----------------\n'
                       f'World size: ({WORLD_SIZE_ROWS}, {WORLD_SIZE_COLS})\n'
                       f'Starting Population: {STARTING_POPULATION}\n'
                       f'Number of Generations: {NUM_GENERATIONS}\n'
                       f'Steps per Generation: {STEPS_PER_GEN}\n'
                       f'Genome start length: {GENOME_START_LENGTH}\n'
                       f'Number of hidden neurons: {HIDDEN_NEURONS}\n'
                       f'Mutation Rate Structural: {MUTATION_RATE_STRUCT}, Non-Structural: {MUTATION_RATE_NON_STRUCT}\n'
                       f'Time to live: {TIME_TO_LIVE}\n'
                       '----------------\n')

        self.log_population()

    def print_monitor(self):
        print("----------------------")
        print(f"Generation {self.generation} Step {self.gen_step + 1}")
        print("----------------------")
        print(f"All-time population: {self.all_time_population}")
        print(f"Current population: {self.total_population}")
        print(f"Number of males: {self.num_males}, Number of females: {self.num_females}")
        print(f"Total successful reproductions: {self.num_reproductions}")
        print(f"Avg. fitness: {self.avg_fitness}")
        print(f"Total number of mutations: {self.total_mutations}")
        print(f"Total number of predators: {self.num_predators}")
        print(f"Organisms killed: {self.num_killed}")
        print(f"Number of species: {self.num_species}")
        print("Species List: ")
        for idx, name in enumerate(self.species_names):
            print(f"{idx+1}. {name}: Pop: {self.species_populations[idx]}, Fitness: {self.species_fitnesses[idx]}")
        print("-----------------------")

    def log_fitness(self):
        with open(self.fitness_file, 'a') as file:
            file.write(str(self.avg_fitness) + '\n')

    def log_population(self):
        with open(self.population_file, 'a') as file:
            file.write(str(self.total_population) + '\n')
