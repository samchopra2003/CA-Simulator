import os

import numpy as np
from dotenv import load_dotenv

from Genome import Genome
from NeuralNetwork import NeuralNetwork

load_dotenv()

TIME_TO_LIVE = int(os.getenv("TIME_TO_LIVE"))


class Organism:
    def __init__(self, location: tuple, time_to_live: int = TIME_TO_LIVE, parents: tuple = None):
        self.location = location
        self.genome = Genome()
        self.nn = NeuralNetwork(self.genome.gene_list)
        # 0 for male, 1 for female
        self.sex = np.random.choice([0, 1])
        self.time_to_live = time_to_live

        # number of time steps
        self.age = 0
        self.fitness = 0

        # historical info
        self.parents = parents
        if self.parents is not None:
            self.fitness = np.average([parents[0].fitness, parents[1].fitness])
        self.children = []
        self.sexual_partners = []

        # 0: Left, 1: Right, 2: Up, 3: Down
        self.last_move = None

        # TODO: Add energy/hunger function

    def step(self, world_state):
        action_probs, output_neuron_ids = self.nn.forward(self, world_state)
        no_action_prob = 1 - np.sum(action_probs)
        action_probs = np.append(action_probs, no_action_prob)
        action_neuron_id = np.random.choice(len(output_neuron_ids)+1, p=action_probs)

        # output neuron actions




