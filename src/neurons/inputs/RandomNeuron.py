import numpy as np
from src.neurons.Neuron import Neuron

neuron_id = 11
neuron_class = 0


class RandomNeuron(Neuron):
    """
    Input Random Neuron
    """

    def __init__(self):
        super().__init__(neu_id=neuron_id, neu_class=neuron_class)

    def forward(self, organism, world_state=None, input_prob=None):
        return np.random.uniform()
