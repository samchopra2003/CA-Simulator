import numpy as np

from src.Organism import Organism
from src.utils.Monitor import Monitor
from src.utils.speciation import segregate_species
from src.utils.update_organism_colors import update_organisms_color


def step(world: np.ndarray, world_state: np.ndarray, organisms: list[Organism], species: dict,
         species_specific_colors=False):
    """
    :param world: World GUI
    :param world_state: Internal world state
    :param organisms: Organism list
    :param species: dict of Species
    :param species_specific_colors: Will display distinct colors on UI for different species
    :return: None
    """
    fitnesses = []
    new_children = []
    num_mutations = 0
    num_predators = 0
    num_killed = 0
    num_males = 0
    num_females = 0
    for org_idx, org in enumerate(organisms):
        if not org.alive:
            continue

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
        org = organisms[idx]
        organisms.pop(idx)
        if org.species in species:
            species[org.species].remove(org)
            if not species[org.species]:  # remove species from dict (species extinction)
                del species[org.species]

    # add new children
    if new_children:
        organisms.extend(new_children)
        segregate_species(new_children, species)

    new_parents_idxs = [idx for idx, org in enumerate(organisms) if org.gave_birth]
    for par_idx in new_parents_idxs:
        organisms[par_idx].gave_birth = False

    # update monitor
    monitor = Monitor()
    monitor.total_population = len(organisms)
    if fitnesses:
        monitor.avg_fitness = np.average(fitnesses)
    else:
        monitor.avg_fitness = 0
    monitor.total_mutations = num_mutations
    monitor.num_killed = num_killed
    monitor.num_predators = num_predators
    monitor.num_reproductions += len(new_children)
    monitor.num_males = num_males
    monitor.num_females = num_females

    # update species data
    monitor.num_species = len(species)
    species_fitnesses = [np.average([org.fitness for org in orgs]) for _, orgs in species.items()]
    species_names = [name for name, _ in species.items()]
    species_populations = [len(orgs) for _, orgs in species.items()]

    if len(species) > 0:
        species_data = list(zip(species_fitnesses, species_names, species_populations))
        sorted_species_data = sorted(species_data, key=lambda x: x[0], reverse=True)
        monitor.species_fitnesses, monitor.species_names, monitor.species_populations = \
            map(list, zip(*sorted_species_data))
    else:
        monitor.species_fitnesses.clear()
        monitor.species_names.clear()
        monitor.species_populations.clear()

    # species specific color
    if species_specific_colors:
        species_names = set(monitor.species_names)
        species_colors_names = set(map(str, monitor.species_colors.keys()))

        if len(species) > len(monitor.species_colors):  # birth of offspring
            monitor.species_colors.update({name: np.random.randint(0, 256, size=3, dtype=np.uint8) for name in
                                           species_names - species_colors_names})
        elif len(species) < len(monitor.species_colors):  # species extinction
            for spec_name in species_colors_names - species_names:
                del monitor.species_colors[spec_name]

        if new_children:
            update_organisms_color(species, monitor.species_colors)

    # log
    monitor.log_fitness()
    monitor.log_population()
