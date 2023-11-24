import os

from dotenv import load_dotenv

from src.neurons.Neuron import Neuron

neuron_id = 7
neuron_class = 0

load_dotenv()

POPULATION_DENSITY_NEIGHBOURHOOD_SIZE = int(os.getenv("POPULATION_DENSITY_NEIGHBOURHOOD_SIZE"))


class PopulationDensityNeuron(Neuron):
    """
    Input Population Density Neuron
    """

    def __init__(self):
        super().__init__(neu_id=neuron_id, neu_class=neuron_class)

    def forward(self, organism, world_state, input_prob=None):
        neighbourhood_size = POPULATION_DENSITY_NEIGHBOURHOOD_SIZE
        row, col = organism.location

        center_x = (neighbourhood_size - 1) // 2
        center_y = (neighbourhood_size - 1) // 2

        row_start = max(row - center_y, 0)
        row_end = min(row + center_y + 1, world_state.shape[0])
        col_start = max(col - center_x, 0)
        col_end = min(col + center_x + 1, world_state.shape[1])

        neighbourhood = world_state[row_start: row_end, col_start: col_end]

        return len(neighbourhood[neighbourhood != 0])
