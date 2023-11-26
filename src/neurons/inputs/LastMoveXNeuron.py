import numpy as np

from src.neurons.Neuron import Neuron

neuron_id = 8
neuron_class = 0


class LastMoveXNeuron(Neuron):
    """
    Input Last Move X Neuron
    0: Right, 1: Left
    """
    def __init__(self):
        super().__init__(neu_id=neuron_id, neu_class=neuron_class)

    def forward(self, organism, world_state: np.ndarray = None, input_prob: float = None):
        if organism.last_move == 0 or organism.last_move == 1:
            return 1
        return 0
