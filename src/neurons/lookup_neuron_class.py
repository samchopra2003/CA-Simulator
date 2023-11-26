import os
from dotenv import load_dotenv

load_dotenv()

INPUT_NEURONS = int(os.getenv("INPUT_NEURONS"))
HIDDEN_NEURONS = int(os.getenv("HIDDEN_NEURONS"))
OUTPUT_NEURONS = int(os.getenv("OUTPUT_NEURONS"))


def lookup_neuron_class(neuron_id: int):
    """
    :param neuron_id: Must be between 1 and 17 + HIDDEN_NEURONS
    :return: neuron class
    """
    # input neurons
    if 1 <= neuron_id <= INPUT_NEURONS:
        return 0
    # output neurons
    elif INPUT_NEURONS + 1 <= neuron_id <= INPUT_NEURONS + OUTPUT_NEURONS:
        return 2
    # hidden neurons
    return 1
