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

    def _init_class_vars(self, organism_list, avg_fitness=0, total_mutations=0, num_predators=0):
        self.organism_list = organism_list
        self.total_population = len(self.organism_list)
        self.avg_fitness = avg_fitness
        self.total_mutations = total_mutations
        self.num_predators = num_predators
