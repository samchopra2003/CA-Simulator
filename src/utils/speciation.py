import os

import numpy as np
from dotenv import load_dotenv

from src.utils.name_generator import generate_species_name

load_dotenv()

COMPATIBILITY_THRESH = float(os.getenv("COMPATIBILITY_THRESH"))
C1 = float(os.getenv("C1"))
C2 = float(os.getenv("C2"))
C3 = float(os.getenv("C3"))


def segregate_species(organisms: list, species=None):
    """
    Segregate new Organisms into distinct species using NEAT-like algorithm.
    :param: List of Organisms to be speciated
    :param: Dict of Species Name and respective List of Organisms
    :return: List of species with Genomes, List of species names
    """
    # TODO: Integrate into species fitness
    if species is None:
        species = []

    species_names = []
    for org in organisms:
        species_found = False
        spec_idx = 0
        for spec_idx, spec in enumerate(species):
            rand_genome_idx = np.random.randint(0, len(spec))

            rep_genome = spec[rand_genome_idx].genome.gene_list  # representative genome

            num_excess_genes = abs(len(rep_genome) - len(org.genome.gene_list))

            gene_set_1 = set(rep_genome)
            gene_set_2 = set(org.genome.gene_list)
            num_disjoint_genes = len(gene_set_1.symmetric_difference(gene_set_2))

            matching_genes_1 = gene_set_1.intersection(gene_set_2)
            matching_genes_2 = gene_set_2.intersection(gene_set_1)

            if matching_genes_1 and matching_genes_2:
                matching_weights_1 = np.array([gene.weight for gene in matching_genes_1])
                matching_weights_2 = np.array([gene.weight for gene in matching_genes_2])
                weight_diff_avg = np.average(np.abs(matching_weights_2 - matching_weights_1))
            else:
                weight_diff_avg = 0.0

            max_genome_len = max(len(rep_genome), len(org.genome.gene_list))

            # compute distance measure
            delta = (C1 * num_excess_genes) / max_genome_len + (C2 * num_disjoint_genes) / max_genome_len + \
                C3 * weight_diff_avg

            if delta < COMPATIBILITY_THRESH:
                species_found = True
                break

        if species_found:
            org.species = species[spec_idx][0].species
            species[spec_idx].append(org)

        else:  # new species
            species.append([org])
            name = generate_species_name()
            org.species = name
            species_names.append(name)

    return species, species_names
