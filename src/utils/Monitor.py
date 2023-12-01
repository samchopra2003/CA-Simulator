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

    def print_monitor(self):
        print(f"All-time population: {self.all_time_population}")
        print(f"Current population: {self.total_population}")
        print(f"Number of males: {self.num_males}, Number of females: {self.num_females}")
        print(f"Total successful reproductions: {self.num_reproductions}")
        print(f"Avg. fitness: {self.avg_fitness}")
        print(f"Total number of mutations: {self.total_mutations}")
        print(f"Total number of predators: {self.num_predators}")
        print(f"Organisms killed: {self.num_killed}")

