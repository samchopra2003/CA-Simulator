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
        species = {}

    for org in organisms:
        species_found = False
        for species_name, species_orgs in species.items():
            rand_genome_idx = np.random.randint(0, len(species_orgs))

            rep_genome = species_orgs[rand_genome_idx].genome.gene_list  # representative genome]

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
                org.species = species_name
                species[species_name].append(org)
                break

        if not species_found:  # new species
            name = generate_species_name()
            org.species = name
            species[name] = [org]

    # return species, species_names
    return species
