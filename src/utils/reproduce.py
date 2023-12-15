import os

import numpy as np
from dotenv import load_dotenv
from src.Genome import Genome

load_dotenv()

WORLD_SIZE_ROWS = int(os.getenv("WORLD_SIZE_ROWS"))
WORLD_SIZE_COLS = int(os.getenv("WORLD_SIZE_COLS"))
CHILD_SPAWN_NEIGHBOURHOOD = int(os.getenv("CHILD_SPAWN_NEIGHBOURHOOD"))


def reproduce(world, world_state, organism):
    """
    Crossover/recombination like NEAT.
    Sexual reproduction
    :param world: World GUI
    :param world_state: Internal world state
    :param organism: Organism list
    :return: partner, position (y, x), color, parents (Organism father, Organism mother), child_genome (Genome)
    """
    # TODO: Fitness should be above a certain threshold
    if organism.fertility > np.random.random() and not organism.gave_birth \
            and organism.age >= organism.min_reproduction_age:
        # search immediate neighbourhood for sexual partner (male and female partners)
        partner = None
        directions = [1, 2, 3, 4]   # right, left, up, down
        directions = np.random.permutation(directions)
        fertilities = None    # tuple: organism fertility, partner fertility

        for cur_dir in directions:
            if cur_dir == 1:
                partner = search_right(world_state, organism)
            elif cur_dir == 2:
                partner = search_left(world_state, organism)
            elif cur_dir == 3:
                partner = search_up(world_state, organism)
            else:
                partner = search_down(world_state, organism)

            if partner:
                if organism.species == partner.species:
                    fertilities = (organism.fertility, partner.fertility)
                else:   # cross-species
                    fertilities = (organism.cross_species_fertility, partner.cross_species_fertility)
                break

        # reproduction criteria
        if partner is None or fertilities[0] < np.random.random() or fertilities[1] < np.random.random() \
                or partner.gave_birth or partner.age < partner.min_reproduction_age:
            return None, None, None, None, None

        # check if nearby vacant spot
        neighbourhood_size = CHILD_SPAWN_NEIGHBOURHOOD
        row, col = organism.position['y'], organism.position['x']

        center_x = (neighbourhood_size - 1) // 2
        center_y = (neighbourhood_size - 1) // 2

        row_start = max(row - center_y, 0)
        row_end = min(row + center_y + 1, world_state.shape[0])
        col_start = max(col - center_x, 0)
        col_end = min(col + center_x + 1, world_state.shape[1])

        neighbourhood = world_state[row_start: row_end, col_start: col_end]
        vacant_pos = np.where(neighbourhood == 0)
        # convert to world_state coordinates
        vacant_pos = (vacant_pos[0] + row_start, vacant_pos[1] + col_start)

        if len(vacant_pos[0]) > 0:
            child_gene_ids = [{"source": gene.source_neuron_id, "sink": gene.sink_neuron_id}
                              for gene in organism.genome.gene_list]
            child_genes = organism.genome.gene_list
            child_gene_owner = [0] * len(organism.genome.gene_list)  # 0 for organism, 1 for partner
            for gene_idx, gene in enumerate(partner.genome.gene_list):
                gene_neu = {"source": gene.source_neuron_id, "sink": gene.sink_neuron_id}
                if gene_neu not in child_gene_ids:
                    child_gene_ids.append(gene_neu)
                    child_genes.append(gene)
                    child_gene_owner.append(1)
                else:
                    # TODO: Fitness of both organisms should be proportional to gene contribution
                    child_gene_idx = child_gene_ids.index(gene_neu)
                    if partner.fitness > organism.fitness:
                        child_genes[child_gene_idx] = gene
                        child_gene_owner[child_gene_idx] = 1
                    elif partner.fitness == organism.fitness:
                        gene_pick_prob = np.random.choice(np.array([0, 1]))
                        if gene_pick_prob == 0:
                            child_genes[child_gene_idx] = gene
                            child_gene_owner[child_gene_idx] = 1

            try:
                vacant_pos_y = vacant_pos[0][0]
                vacant_pos_x = vacant_pos[1][0]

                color = np.random.randint(0, 256, size=3, dtype=np.uint8)
                pos = (vacant_pos_y, vacant_pos_x)
                world[pos] = color
                if organism.sex == 0:
                    parents = (organism, partner)
                else:
                    parents = (partner, organism)
                # TODO: Implement fitness
                child_genome = Genome(gene_list=child_genes)
                return partner, pos, color, parents, child_genome
            except IndexError:
                pass

    return None, None, None, None, None


def search_right(world_state, organism):
    try:
        if world_state[organism.position['y'], organism.position['x'] + 1] != 0:  # right
            if world_state[organism.position['y'], organism.position['x'] + 1].sex != organism.sex:
                return world_state[organism.position['y'], organism.position['x'] + 1]
    except IndexError:
        return None


def search_left(world_state, organism):
    try:
        if world_state[organism.position['y'], organism.position['x'] - 1] != 0:  # left
            if world_state[organism.position['y'], organism.position['x'] - 1].sex != organism.sex:
                return world_state[organism.position['y'], organism.position['x'] - 1]
    except IndexError:
        return None


def search_up(world_state, organism):
    try:
        if world_state[organism.position['y'] - 1, organism.position['x']] != 0:  # up
            if world_state[organism.position['y'] - 1, organism.position['x']].sex != organism.sex:
                return world_state[organism.position['y'] - 1, organism.position['x']]
    except IndexError:
        return None


def search_down(world_state, organism):
    try:
        if world_state[organism.position['y'] + 1, organism.position['x']] != 0:  # down
            if world_state[organism.position['y'] + 1, organism.position['x']].sex != organism.sex:
                return world_state[organism.position['y'] + 1, organism.position['x']]
    except IndexError:
        return None
