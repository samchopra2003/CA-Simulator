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

    def __init__(self, neuron_id):
        super().__init__(neu_id=neuron_id, neu_class=neuron_class)

    def forward(self, organism, world_state=None, input_prob=None):
        return input_prob
