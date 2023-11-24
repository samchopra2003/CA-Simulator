class Neuron:
    """
    Base Neuron Class
    """

    def __init__(self, neu_id, neu_class):
        self.neuron_id = neu_id
        # 0 = input neuron, 1 = hidden neuron, 2 = output neuron
        self.neuron_class = neu_class

    def forward(self, organism, world_state, input_prob):
        pass
