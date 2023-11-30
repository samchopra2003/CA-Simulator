class Monitor:
    """
    Singleton Monitor Class
    """
    _instance = None

    def __new__(cls, *args):
        if cls._instance is None:
            cls._instance = super(Monitor, cls).__new__(cls)
            cls._instance._init_class_vars(*args)
        return cls._instance

    def _init_class_vars(self, organism_list=None, avg_fitness=0, total_mutations=0, num_predators=0):
        if organism_list is None:
            organism_list = []
        self.organism_list = organism_list
        self.total_population = len(self.organism_list)
        self.avg_fitness = avg_fitness
        self.total_mutations = total_mutations
        self.num_predators = num_predators

    def print_monitor(self):
        print(f"Total population: {self.total_population}")
        print(f"Avg. fitness: {self.avg_fitness}")
        print(f"Total number of mutations: {self.total_mutations}")
        print(f"Total number of predators: {self.num_predators}")
