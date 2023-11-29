import numpy as np

import os
from dotenv import load_dotenv

load_dotenv()

WEIGHT_INIT_LOWER = int(os.getenv("WEIGHT_INIT_LOWER"))
WEIGHT_INIT_UPPER = int(os.getenv("WEIGHT_INIT_UPPER"))


class Gene:
    """
    Connection between two neurons in the Neural Network
    """

    def __init__(self, source_neu_id: int, sink_neu_id: int,
                 weight: float = np.random.uniform(WEIGHT_INIT_LOWER, WEIGHT_INIT_UPPER)):
        self.source_neuron_id = source_neu_id
        self.sink_neuron_id = sink_neu_id
        self.weight = weight
        # TODO: Maybe add bias var

    def init_new_weight(self):
        """
        Initializes new weight due to non-structural mutation
        :return: None
        """
        self.weight = np.random.uniform(WEIGHT_INIT_LOWER, WEIGHT_INIT_UPPER)
