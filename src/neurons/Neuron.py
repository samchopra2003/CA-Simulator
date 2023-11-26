import numpy as np


class Neuron:
    """
    Base Neuron Class
    """

    def __init__(self, neu_id: int, neu_class: int):
        self.neuron_id = neu_id
        # 0 = input neuron, 1 = hidden neuron, 2 = output neuron
        self.neuron_class = neu_class

    def forward(self, organism, world_state: np.ndarray, input_prob: float):
        pass
