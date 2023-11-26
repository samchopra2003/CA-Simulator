import os

import numpy as np
from dotenv import load_dotenv

from src.neurons.Neuron import Neuron

load_dotenv()

WORLD_SIZE_ROWS = int(os.getenv("WORLD_SIZE_ROWS"))

neuron_id = 3
neuron_class = 0


class LocationYNeuron(Neuron):
    """
    Input Location Y Neuron
    """

    def __init__(self):
        super().__init__(neu_id=neuron_id, neu_class=neuron_class)

    def forward(self, organism, world_state: np.ndarray = None, input_prob: float = None):
        return organism.position['y'] / WORLD_SIZE_ROWS
