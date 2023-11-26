import numpy as np

from src.neurons.Neuron import Neuron

neuron_class = 2


class OutputNeuron(Neuron):
    """
    Output Neurons:

    12. Move left/right
    13. Move up/down
    14. Move random
    15. Move forward
    16. Move reverse
    17. Kill forward neighbour
    """

    def __init__(self, neuron_id: int, binary_prob: float = np.random.random()):
        super().__init__(neu_id=neuron_id, neu_class=neuron_class)
        # binary probability for e.g. to move left or right (not going to be used for all out neurons)
        self.binary_probability = binary_prob

    def forward(self, organism, world_state: np.ndarray = None, input_prob: float = None):
        return input_prob
