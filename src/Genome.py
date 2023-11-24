import numpy as np

import os
from dotenv import load_dotenv

from Gene import Gene

load_dotenv()

GENOME_START_LENGTH = int(os.getenv("GENOME_START_LENGTH"))

INPUT_NEURONS = int(os.getenv("INPUT_NEURONS"))
HIDDEN_NEURONS = int(os.getenv("HIDDEN_NEURONS"))
OUTPUT_NEURONS = int(os.getenv("OUTPUT_NEURONS"))


class Genome:
    def __init__(self):
        self.gene_list = []

        self._build_genome()

    def _build_genome(self):
        # TODO: No self connections and same class connections (e.g. no input to input conn) permitted right now
        total_neurons = np.sum([INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS])
        for _ in range(GENOME_START_LENGTH):
            neuron_1_id = np.random.uniform(0, total_neurons)
            if neuron_1_id <= INPUT_NEURONS:  # Neuron 1 Input
                neuron_2_id = np.random.uniform(INPUT_NEURONS, total_neurons)
                source_id = neuron_1_id
                sink_id = neuron_2_id

            elif neuron_1_id <= INPUT_NEURONS + OUTPUT_NEURONS:  # Neuron 1 Output
                neuron_2_id = np.max([np.random.uniform(INPUT_NEURONS + OUTPUT_NEURONS, total_neurons),
                                      np.random.uniform(0, INPUT_NEURONS)])
                source_id = neuron_2_id
                sink_id = neuron_1_id

            else:  # Neuron 1 Hidden
                neuron_2_id = np.random.uniform(0, INPUT_NEURONS + OUTPUT_NEURONS)
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
