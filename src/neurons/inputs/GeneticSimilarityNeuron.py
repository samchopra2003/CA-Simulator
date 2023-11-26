import os

import numpy as np
from dotenv import load_dotenv

from src.neurons.Neuron import Neuron

neuron_id = 4
neuron_class = 0

load_dotenv()

GENETIC_SIMILARITY_NEIGHBOURHOOD_SIZE = int(os.getenv("GENETIC_SIMILARITY_NEIGHBOURHOOD_SIZE"))


class GeneticSimilarityNeuron(Neuron):
    """
    Input Genetic Similarity Neuron
    Genetic similarity based on just number of similar genes / total genes
    Disregard synaptic weights
    """

    def __init__(self):
        super().__init__(neu_id=neuron_id, neu_class=neuron_class)

    def forward(self, organism, world_state: np.ndarray, input_prob: float = None):
        neighbourhood_size = GENETIC_SIMILARITY_NEIGHBOURHOOD_SIZE
        row, col = organism.position['y'], organism.position['x']

        center_x = (neighbourhood_size - 1) // 2
        center_y = (neighbourhood_size - 1) // 2

        row_start = max(row - center_y, 0)
        row_end = min(row + center_y + 1, world_state.shape[0])
        col_start = max(col - center_x, 0)
        col_end = min(col + center_x + 1, world_state.shape[1])

        neighbourhood = world_state[row_start: row_end, col_start: col_end]
        neighbourhood[center_x, center_y] = 0  # disregard current organism

        # extract organisms in neighbourhood
        all_other_organisms = neighbourhood[neighbourhood != 0]
        genetic_similarity_scores = []
        for org in all_other_organisms:
            genetic_similarity_score = 0
            for org_gene in organism.genome.gene_list:
                for other_org_gene in org.genome.gene_list:
                    if org_gene.source_neuron_id == other_org_gene.source_neuron_id and org_gene.sink_neuron_id \
                            == other_org_gene.sink_neuron_id:
                        genetic_similarity_score += 1
                        break

            genetic_similarity_scores.append(genetic_similarity_score / (len(organism.genome.gene_list) + 1.e-10))

        if len(genetic_similarity_scores) == 0:
            return 0
        return np.average(genetic_similarity_scores)
