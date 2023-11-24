import os

from dotenv import load_dotenv

from src.neurons.Neuron import Neuron

neuron_id = 10
neuron_class = 0

load_dotenv()

PREDATOR_DETECTION_NEIGHBOURHOOD_SIZE = int(os.getenv("PREDATOR_DETECTION_NEIGHBOURHOOD_SIZE"))


class PredatorDetectionNeuron(Neuron):
    """
    Input Predator Detection Neuron
    """

    def __init__(self):
        super().__init__(neu_id=neuron_id, neu_class=neuron_class)

    def forward(self, organism, world_state, input_prob=None):
        neighbourhood_size = PREDATOR_DETECTION_NEIGHBOURHOOD_SIZE
        row, col = organism.location

        center_x = (neighbourhood_size - 1) // 2
        center_y = (neighbourhood_size - 1) // 2

        row_start = max(row - center_y, 0)
        row_end = min(row + center_y + 1, world_state.shape[0])
        col_start = max(col - center_x, 0)
        col_end = min(col + center_x + 1, world_state.shape[1])

        neighbourhood = world_state[row_start: row_end, col_start: col_end]
        neighbourhood[center_x, center_y] = 0  # disregard current organism

        # extract organisms in neighbourhood
        all_other_organisms = neighbourhood[neighbourhood != 0]
        kill_neuron_id = 17
        predator_fitnesses = []
        for org in all_other_organisms:
            for org_neu_id in org.nn.enabled_neurons:
                if org_neu_id == kill_neuron_id:
                    predator_fitnesses.append(org.fitness)
                    break

        # predators present in neighbourhood and better fitness than organism
        if len(predator_fitnesses) > 0 and max(predator_fitnesses) > organism.fitness:
            return 1
        return 0
