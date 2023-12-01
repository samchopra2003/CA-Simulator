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
    num_mutations = 0
    num_predators = 0
    num_killed = 0
    for org_idx, org in enumerate(organisms):
        org.step(world, world_state)

        fitnesses.append(org.fitness)
        num_mutations += org.genome.num_mutations
        num_killed += org.kills
        if org.kills >= 1:
            num_predators += 1

    # remove dead organisms
    dead_org_idxs = [idx for idx, org in enumerate(organisms) if not org.alive]
    for idx in reversed(dead_org_idxs):
        organisms.pop(idx)

    # update monitor
    monitor = Monitor()
    monitor.total_population = len(organisms)
    monitor.avg_fitness = np.average(fitnesses)
    monitor.total_mutations = num_mutations
    monitor.num_killed = num_killed
    monitor.num_predators = num_predators
