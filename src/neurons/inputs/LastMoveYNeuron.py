from src.neurons.Neuron import Neuron

neuron_id = 9
neuron_class = 0


class LastMoveYNeuron(Neuron):
    """
    Input Last Move Y Neuron
    2: Up, 3: Down
    """
    def __init__(self):
        super().__init__(neu_id=neuron_id, neu_class=neuron_class)

    def forward(self, organism, world_state=None, input_prob=None):
        return (organism.last_move - 1) * 0.5
