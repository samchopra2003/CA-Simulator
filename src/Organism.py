import os

import numpy as np
from dotenv import load_dotenv
import uuid

from Genome import Genome
from NeuralNetwork import NeuralNetwork
from neurons.Neuron import Neuron
from utils.move import move_right, move_left, move_up, move_down, die

load_dotenv()

TIME_TO_LIVE = int(os.getenv("TIME_TO_LIVE"))


class Organism:
    def __init__(self, position: tuple, color: np.ndarray, time_to_live: int = TIME_TO_LIVE, parents: tuple = None,
                 sex: int = np.random.choice([0, 1]), fitness: float = 0):
        """
        :param position: Position (y, x)
        :param color: Color in GUI
        :param time_to_live: natural life duration
        :param parents: Organism parents
        :param sex: 0 for male, 1 for female
        :param fitness: initial fitness value
        """
        self.id = uuid.uuid4()
        self.position = {'x': position[1], 'y': position[0]}
        self.color = color

        self.genome = Genome()
        self.nn = NeuralNetwork(self.genome.gene_list)

        # number of time steps
        self.age = 0
        self.sex = sex
        self.time_to_live = time_to_live
        self.fitness = fitness
        self.alive = True

        # historical info
        self.parents = parents
        if self.parents is not None:
            self.fitness = np.average([parents[0].fitness, parents[1].fitness])
        self.children = []
        self.sexual_partners = []

        # 0: Right, 1: Left, 2: Up, 3: Down
        self.last_move = None

        # TODO: Add energy/hunger function

    def step(self, world: np.ndarray, world_state: np.ndarray):
        """
        :param world: World GUI
        :param world_state: internal world state
        :return: None
        """
        action_probs, output_neurons = self.nn.forward(self, world_state)
        no_action_prob = 1 - np.sum(action_probs)
        action_probs = np.append(action_probs, no_action_prob)
        # consider blank neuron the do nothing action
        output_neurons.append(Neuron(-1, -1))
        action_neuron = np.random.choice(np.array(output_neurons), p=action_probs)

        # map to output neuron actions
        if action_neuron.neuron_id == 12:  # Move left/right
            if np.random.random() < action_neuron.binary_probability:  # move right
                move_right(world, world_state, self)
            else:  # move left
                move_left(world, world_state, self)

        elif action_neuron.neuron_id == 13:  # Move up/down
            if np.random.random() < action_neuron.binary_probability:  # move up
                move_up(world, world_state, self)
            else:  # move down
                move_down(world, world_state, self)

        elif action_neuron.neuron_id == 14:  # Move random
            move = np.random.choice(4)
            if move == 0:
                move_right(world, world_state, self)
            elif move == 1:
                move_left(world, world_state, self)
            elif move == 2:
                move_up(world, world_state, self)
            else:
                move_down(world, world_state, self)

        elif action_neuron.neuron_id == 15:  # Move forward
            if self.last_move == 0:
                move_right(world, world_state, self)
            elif self.last_move == 1:
                move_left(world, world_state, self)
            elif self.last_move == 2:
                move_up(world, world_state, self)
            else:
                move_down(world, world_state, self)

        elif action_neuron.neuron_id == 16:  # Move reverse
            if self.last_move == 0:
                move_left(world, world_state, self)
            elif self.last_move == 1:
                move_right(world, world_state, self)
            elif self.last_move == 2:
                move_down(world, world_state, self)
            else:
                move_up(world, world_state, self)

        elif action_neuron.neuron_id == 17:  # Kill forward neighbour
            # TODO: Implement kill function
            pass

        self.age += 1
        if self.age == self.time_to_live:   # death
            die(world, world_state, self)
