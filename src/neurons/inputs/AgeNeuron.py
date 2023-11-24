import os

from dotenv import load_dotenv

from src.neurons.Neuron import Neuron

load_dotenv()

TIME_TO_LIVE = int(os.getenv("TIME_TO_LIVE"))

neuron_id = 1
neuron_class = 0


class AgeNeuron(Neuron):
    """
    Input Age Neuron
    """
    def __init__(self):
        super().__init__(neu_id=neuron_id, neu_class=neuron_class)

    def forward(self, organism, world_state=None, input_prob=None):
        return organism.age / TIME_TO_LIVE
