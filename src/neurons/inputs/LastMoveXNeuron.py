from src.neurons.Neuron import Neuron

neuron_id = 8
neuron_class = 0


class LastMoveXNeuron(Neuron):
    """
    Input Last Move X Neuron
    0: Left, 1: Right
    """
    def __init__(self):
        super().__init__(neu_id=neuron_id, neu_class=neuron_class)

    def forward(self, organism, world_state=None, input_prob=None):
        return (organism.last_move + 1) * 0.5
