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
    new_children = []
    new_parents_idxs = []
    num_mutations = 0
    num_predators = 0
    num_killed = 0
    num_males = 0
    num_females = 0
    for org_idx, org in enumerate(organisms):
        org.step(world, world_state)

        if org.sex == 0:
            num_males += 1
        elif org.sex == 1:
            num_females += 1

        fitnesses.append(org.fitness)
        num_mutations += org.genome.num_mutations
        num_killed += org.kills
        if org.kills >= 1:
            num_predators += 1

        if org.gave_birth:
            if org.children[-1] not in new_children and org.children[-1] not in organisms:
                new_children.append(org.children[-1])

    # remove dead organisms
    dead_org_idxs = [idx for idx, org in enumerate(organisms) if not org.alive]
    for idx in reversed(dead_org_idxs):
        organisms.pop(idx)

    # add new children
    organisms.extend(new_children)

    for par_idx in new_parents_idxs:
        organisms[par_idx].gave_birth = False

    # update monitor
    monitor = Monitor()
    monitor.total_population = len(organisms)
    monitor.avg_fitness = np.average(fitnesses)
    monitor.total_mutations = num_mutations
    monitor.num_killed = num_killed
    monitor.num_predators = num_predators
    monitor.num_reproductions += len(new_children)
    monitor.num_males = num_males
    monitor.num_females = num_females

    # log
    monitor.log_fitness()
    monitor.log_population()
