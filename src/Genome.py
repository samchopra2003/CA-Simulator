import numpy as np

import os
from dotenv import load_dotenv

from Gene import Gene
from neurons.lookup_neuron_class import lookup_neuron_class

load_dotenv()

GENOME_START_LENGTH = int(os.getenv("GENOME_START_LENGTH"))

INPUT_NEURONS = int(os.getenv("INPUT_NEURONS"))
HIDDEN_NEURONS = int(os.getenv("HIDDEN_NEURONS"))
OUTPUT_NEURONS = int(os.getenv("OUTPUT_NEURONS"))

MUTATION_RATE_STRUCT = float(os.getenv("MUTATION_RATE_STRUCT"))
MUTATION_RATE_NON_STRUCT = float(os.getenv("MUTATION_RATE_STRUCT"))

WEIGHT_INIT_LOWER = int(os.getenv("WEIGHT_INIT_LOWER"))
WEIGHT_INIT_UPPER = int(os.getenv("WEIGHT_INIT_UPPER"))
WEIGHT_DELTA = int(os.getenv("WEIGHT_DELTA"))

TOTAL_NEURONS = np.sum([INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS])


class Genome:
    def __init__(self):
        self.gene_list = []

        self._build_genome()

        self._mutate()

    def _build_genome(self):
        # TODO: No self connections and same class connections (e.g. no input to input conn) permitted right now
        for _ in range(GENOME_START_LENGTH):
            self._build_gene()

    def _build_gene(self):
        neuron_1_id = np.random.randint(0, TOTAL_NEURONS)
        if neuron_1_id <= INPUT_NEURONS:  # Neuron 1 Input
            neuron_2_id = np.random.randint(INPUT_NEURONS, TOTAL_NEURONS)
            source_id = neuron_1_id
            sink_id = neuron_2_id

        elif neuron_1_id <= INPUT_NEURONS + OUTPUT_NEURONS:  # Neuron 1 Output
            neuron_2_id = np.max([np.random.randint(INPUT_NEURONS + OUTPUT_NEURONS, TOTAL_NEURONS),
                                  np.random.randint(0, INPUT_NEURONS)])
            source_id = neuron_2_id
            sink_id = neuron_1_id

        else:  # Neuron 1 Hidden
            neuron_2_id = np.random.randint(0, INPUT_NEURONS + OUTPUT_NEURONS)
            if neuron_2_id <= INPUT_NEURONS:
                source_id = neuron_2_id
                sink_id = neuron_1_id
            else:
                source_id = neuron_1_id
                sink_id = neuron_2_id

        # no duplicate genes
        for genes in self.gene_list:
            if genes.source_neuron_id == source_id and genes.sink_neuron_id == sink_id:
                continue

        self.gene_list.append(Gene(source_id, sink_id))

    def _mutate(self):
        """
        Applies structural and/or non-structural mutation to the gene.
        :return: None
        """
        # Structural mutation
        if np.random.random() < MUTATION_RATE_STRUCT:
            # 0 for addition of synapse, 1 for removal of synapse
            # 2 for addition of hidden neuron, 3 for removal of hidden neuron
            mutation_type = np.random.choice(np.array([0, 1, 2, 3]))
            if mutation_type == 0:  # addition of synapse/gene
                self._build_gene()
            elif mutation_type == 1:  # removal of synapse/gene
                removed_gene_idx = np.random.randint(0, len(self.gene_list))
                self.gene_list.pop(removed_gene_idx)
            elif mutation_type == 2:  # addition of hidden neuron to existing synapse/gene (create new gene)
                mutated_gene_idx = np.random.randint(0, len(self.gene_list))
                mutated_gene = self.gene_list[mutated_gene_idx]

                # input and output synapse
                if lookup_neuron_class(mutated_gene.source_neuron_id) == 0 and \
                        lookup_neuron_class(mutated_gene.sink_neuron_id) == 2:
                    conn_neu = np.random.choice(np.array([0, 1]))
                    hidden_neu_id = np.random.choice(np.arange(INPUT_NEURONS + OUTPUT_NEURONS, TOTAL_NEURONS))
                    if conn_neu == 0:  # create hidden neuron and input neuron gene
                        self.gene_list.append(Gene(mutated_gene.source_neuron_id, hidden_neu_id))
                    else:  # create hidden neuron and output neuron gene
                        self.gene_list.append(Gene(hidden_neu_id, mutated_gene.sink_neuron_id))

                # input and hidden synapse
                elif lookup_neuron_class(mutated_gene.source_neuron_id) == 0 and \
                        lookup_neuron_class(mutated_gene.sink_neuron_id) == 1:
                    hidden_neu_id = np.random.choice(np.arange(INPUT_NEURONS + OUTPUT_NEURONS, TOTAL_NEURONS))
                    self.gene_list.append(Gene(mutated_gene.source_neuron_id, hidden_neu_id))

                # hidden and output synapse
                elif lookup_neuron_class(mutated_gene.source_neuron_id) == 1 and \
                        lookup_neuron_class(mutated_gene.sink_neuron_id) == 2:
                    hidden_neu_id = np.random.choice(np.arange(INPUT_NEURONS + OUTPUT_NEURONS, TOTAL_NEURONS))
                    self.gene_list.append(Gene(hidden_neu_id, mutated_gene.sink_neuron_id))

            else:  # removal of hidden neuron (remove all genes containing the hidden neuron)
                permute_idxs = np.random.permutation(len(self.gene_list))
                hidden_neuron_id = -1
                removed_idxs = []
                for gene_idx in permute_idxs:
                    source_id = self.gene_list[gene_idx].source_neuron_id
                    sink_id = self.gene_list[gene_idx].sink_neuron_id
                    if (source_id == hidden_neuron_id or sink_id == hidden_neuron_id) and hidden_neuron_id != -1:
                        removed_idxs.append(gene_idx)
                    elif source_id == 1 and hidden_neuron_id == -1:
                        hidden_neuron_id = source_id
                    elif sink_id == 1 and hidden_neuron_id == -1:
                        hidden_neuron_id = sink_id

                self.gene_list = \
                    [item for idx, item in enumerate(self.gene_list) if idx not in removed_idxs]

        # Non-structural mutation
        if np.random.random() < MUTATION_RATE_NON_STRUCT:
            # 0 for delta to an existing random synaptic weight
            # 1 for a completely new random synaptic weight
            mutation_type = np.random.choice(np.array([0, 1]))
            mutated_gene_idx = np.random.randint(0, len(self.gene_list))
            if mutation_type == 0:  # change synaptic weight
                weight_delta = np.random.uniform(WEIGHT_INIT_LOWER + WEIGHT_DELTA, WEIGHT_INIT_UPPER - WEIGHT_DELTA)
                self.gene_list[mutated_gene_idx].weight += weight_delta
            else:  # new synaptic weight
                self.gene_list[mutated_gene_idx].init_new_weight()
