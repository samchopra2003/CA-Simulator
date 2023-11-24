import os

import numpy as np
from dotenv import load_dotenv

from src.neurons.create_neuron import create_neuron
from src.neurons.lookup_neuron_class import lookup_neuron_class
from src.math_utils.softmax import softmax

load_dotenv()

INPUT_NEURONS = int(os.getenv("INPUT_NEURONS"))
HIDDEN_NEURONS = int(os.getenv("HIDDEN_NEURONS"))
OUTPUT_NEURONS = int(os.getenv("OUTPUT_NEURONS"))


class NeuralNetwork:
    def __init__(self, gene_list):
        total_neurons = np.sum([INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS])
        self.adjacency_matrix = np.zeros((total_neurons, total_neurons))
        # TODO: Explained in Genome.py
        # # source neuron, sink neuron
        # self.synapses = []
        self.enabled_neurons = []
        # reserved for hidden and output neurons
        self.neuron_inputs = {}

        # different class synaptic connections (removes need for sorting)
        # source neuron, sink neuron
        self.input_source_synapses = []
        self.hidden_source_synapses = []

        self._build_brain_wiring(gene_list)
        self._cull_useless_neurons()

    def _build_brain_wiring(self, gene_list):
        for gene in gene_list:
            self.adjacency_matrix[gene.source_neuron_id][gene.sink_neuron_id] = 1
            self.adjacency_matrix[gene.sink_neuron_id][gene.source_neuron_id] = 1

            # instantiate neurons
            source_neuron = create_neuron(gene.source_neuron_id)
            sink_neuron = create_neuron(gene.sink_neuron_id)

            # sink neuron has to be either hidden or output in current config
            if gene.sink_neuron_id not in self.neuron_inputs:
                self.neuron_inputs[gene.sink_neuron_id] = []

            # self.synapses.append([source_neuron, sink_neuron])
            if source_neuron.neuron_class == 0:  # Input neuron
                self.input_source_synapses.append([source_neuron, sink_neuron])
            else:  # Hidden neuron
                self.hidden_source_synapses.append([source_neuron, sink_neuron])

            self.enabled_neurons.append(gene.source_neuron_id)
            self.enabled_neurons.append(gene.sink_neuron_id)

    def _cull_useless_neurons(self):
        """
        Cull useless neurons and associated connections
        Hidden layer connected to no input or no output neurons
        """
        if len(self.input_source_synapses) > 0:
            # check if hidden layer has no output neurons
            removed_synapse_idx = []
            for syn_idx, syn in enumerate(self.input_source_synapses):
                sink_neuron = syn[0]
                output_found = False
                for col_idx in range(self.adjacency_matrix.shape[1]):
                    if self.adjacency_matrix[sink_neuron.neuron_id][col_idx] == 1:
                        if lookup_neuron_class(col_idx) == 2:
                            output_found = True
                            break

                if not output_found:
                    removed_synapse_idx.append(syn_idx)
                    self.adjacency_matrix[sink_neuron.neuron_id][syn[1]] = 0
                    continue

            self.input_source_synapses = \
                [item for idx, item in enumerate(self.input_source_synapses) if idx not in removed_synapse_idx]
            # self.hidden_source_synapses should be able to remove appropriate synaptic connections for
            # troublesome hidden neuron

        if len(self.hidden_source_synapses) > 0:
            # check if hidden layer has no input neurons
            removed_synapse_idx = []
            for syn_idx, syn in enumerate(self.hidden_source_synapses):
                source_neuron = syn[0]
                input_found = False
                for col_idx in range(self.adjacency_matrix.shape[1]):
                    if self.adjacency_matrix[source_neuron.neuron_id][col_idx] == 1:
                        if lookup_neuron_class(col_idx) == 0:
                            input_found = True
                            break

                if not input_found:
                    removed_synapse_idx.append(syn_idx)
                    continue

            self.hidden_source_synapses = \
                [item for idx, item in enumerate(self.hidden_source_synapses) if idx not in removed_synapse_idx]

    def forward(self, organism, world_state):
        """
        Forward Pass of Neural Network
        :return: Action Probabilities from Output Neurons
        """
        if len(self.input_source_synapses) > 0:  # Evaluate Input Neurons
            for syn in self.input_source_synapses:
                source_neuron = syn[0]
                sink_neuron = syn[1]
                input_out = source_neuron.forward(organism, world_state, input_prob=None)
                self.neuron_inputs[sink_neuron].append(input_out)

        outputs = []
        output_neuron_ids = []
        if len(self.hidden_source_synapses) > 0:  # Evaluate Hidden Neurons
            for syn in self.hidden_source_synapses:
                source_neuron = syn[0]
                sink_neuron = syn[1]
                # tanh Activation Function
                input_prob = np.tanh(np.sum(self.neuron_inputs[source_neuron.neuron_id]))
                hidden_out = source_neuron.forward(organism, world_state, input_prob=input_prob)
                self.neuron_inputs[sink_neuron].append(hidden_out)

            # Evaluate Output Neurons
            for syn in self.hidden_source_synapses:
                source_neuron = syn[0]
                sink_neuron = syn[1]
                # tanh Activation Function
                input_prob = np.tanh(np.sum(self.neuron_inputs[source_neuron.neuron_id]))
                out = source_neuron.forward(organism, world_state, input_prob=input_prob)
                outputs.append(out)
                output_neuron_ids.append(sink_neuron.neuron_id)

        self.neuron_inputs.clear()
        return softmax(outputs), output_neuron_ids
