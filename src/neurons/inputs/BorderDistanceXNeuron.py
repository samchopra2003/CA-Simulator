import os

from dotenv import load_dotenv

from src.neurons.Neuron import Neuron

load_dotenv()

WORLD_SIZE_COLS = int(os.getenv("WORLD_SIZE_COLS"))

neuron_id = 5
neuron_class = 0


class BorderDistanceXNeuron(Neuron):
    """
    Border Distance X Neuron
    """

    def __init__(self):
        super().__init__(neu_id=neuron_id, neu_class=neuron_class)

    def forward(self, organism, world_state=None, input_prob=None):
        return min(abs(WORLD_SIZE_COLS - organism.location[1]), organism.location[1])
