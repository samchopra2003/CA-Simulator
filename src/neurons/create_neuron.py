import os

from dotenv import load_dotenv

from src.neurons.hidden.HiddenNeuron import HiddenNeuron
from src.neurons.inputs.AgeNeuron import AgeNeuron
from src.neurons.inputs.BorderDistanceXNeuron import BorderDistanceXNeuron
from src.neurons.inputs.BorderDistanceYNeuron import BorderDistanceYNeuron
from src.neurons.inputs.GeneticSimilarityNeuron import GeneticSimilarityNeuron
from src.neurons.inputs.LastMoveXNeuron import LastMoveXNeuron
from src.neurons.inputs.LastMoveYNeuron import LastMoveYNeuron
from src.neurons.inputs.LocationXNeuron import LocationXNeuron
from src.neurons.inputs.LocationYNeuron import LocationYNeuron
from src.neurons.inputs.PopulationDensityNeuron import PopulationDensityNeuron
from src.neurons.inputs.PredatorDetectionNeuron import PredatorDetectionNeuron
from src.neurons.inputs.RandomNeuron import RandomNeuron
from src.neurons.outputs.OutputNeuron import OutputNeuron

load_dotenv()

INPUT_NEURONS = int(os.getenv("INPUT_NEURONS"))
OUTPUT_NEURONS = int(os.getenv("OUTPUT_NEURONS"))


def create_neuron(neuron_id: int):
    """
    :param neuron_id: Must be between 1 and 17 + HIDDEN_NEURONS
    :return: Neuron
    """
    # input neurons
    if neuron_id == 1:
        return AgeNeuron()
    elif neuron_id == 2:
        return LocationXNeuron()
    elif neuron_id == 3:
        return LocationYNeuron()
    elif neuron_id == 4:
        return GeneticSimilarityNeuron()
    elif neuron_id == 5:
        return BorderDistanceXNeuron()
    elif neuron_id == 6:
        return BorderDistanceYNeuron()
    elif neuron_id == 7:
        return PopulationDensityNeuron()
    elif neuron_id == 8:
        return LastMoveXNeuron()
    elif neuron_id == 9:
        return LastMoveYNeuron()
    elif neuron_id == 10:
        return PredatorDetectionNeuron()
    elif neuron_id == 11:
        return RandomNeuron()
    # output neurons
    elif INPUT_NEURONS + 1 <= neuron_id <= INPUT_NEURONS + OUTPUT_NEURONS:
        return OutputNeuron(neuron_id)
    # hidden neurons
    return HiddenNeuron(neuron_id)
