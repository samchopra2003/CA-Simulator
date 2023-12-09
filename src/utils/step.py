import numpy as np

from src.Organism import Organism
from src.utils.Monitor import Monitor
from src.utils.speciation import segregate_species


def step(world: np.ndarray, world_state: np.ndarray, organisms: list[Organism], species: dict):
    """
    :param world: World GUI
    :param world_state: Internal world state
    :param organisms: Organism list
    :param species: dict of Species
    :return: None
    """
    fitnesses = []
    new_children = []
    num_mutations = 0
    num_predators = 0
    num_killed = 0
    num_males = 0
    num_females = 0
    population_change = False
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
            population_change = True

        if org.gave_birth:
            if org.children[-1] not in new_children and org.children[-1] not in organisms:
                new_children.append(org.children[-1])
            population_change = True

    # remove dead organisms
    dead_org_idxs = [idx for idx, org in enumerate(organisms) if not org.alive]
    for idx in reversed(dead_org_idxs):
        org = organisms[idx]
        organisms.pop(idx)
        if org in species[org.species]:
            species[org.species].remove(org)
            if not species[org.species]:    # remove species from list (species extinction)
                del species[org.species]

    # add new children
    organisms.extend(new_children)
    species = segregate_species(new_children, species)

    new_parents_idxs = [idx for idx, org in enumerate(organisms) if org.gave_birth]
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

    # update species data
    if population_change:
        monitor.num_species = len(species)
        species_fitnesses = [np.average([org.fitness for org in orgs]) for _, orgs in species.items()]
        species_names = [name for name, _ in species.items()]
        species_populations = [len(orgs) for _, orgs in species.items()]

        species_data = list(zip(species_fitnesses, species_names, species_populations))
        sorted_species_data = sorted(species_data, key=lambda x: x[0], reverse=True)
        monitor.species_fitnesses, monitor.species_names, monitor.species_populations = zip(*sorted_species_data)

    # log
    monitor.log_fitness()
    monitor.log_population()
