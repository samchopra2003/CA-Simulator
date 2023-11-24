from src.neurons.Neuron import Neuron

neuron_class = 1


class HiddenNeuron(Neuron):
    """
    Hidden Neuron Class
    """

    def __init__(self, neuron_id):
        super().__init__(neu_id=neuron_id, neu_class=neuron_class)

    def forward(self, organism, world_state=None, input_prob=None):
        return input_prob
