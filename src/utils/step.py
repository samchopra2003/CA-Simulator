import numpy as np

from src.Organism import Organism
from src.utils.Monitor import Monitor


def step(world: np.ndarray, world_state: np.ndarray, organisms: list[Organism]):
    """
    :param world: World GUI
    :param world_state: Internal world state
    :param organisms: Organism list
    :return: None
    """
    fitnesses = []
    for org in organisms:
        # TODO: Kill functionality not created yet
        org.step(world, world_state)
        fitnesses.append(org.fitness)

    # update monitor
    monitor = Monitor()
    monitor.organism_list = organisms
    monitor.total_population = len(organisms)
    monitor.avg_fitness = np.average(fitnesses)
