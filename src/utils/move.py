import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()
WORLD_BACKGROUND_COLOR = int(os.getenv("WORLD_BACKGROUND_COLOR"))


def move_right(world: np.ndarray, world_state: np.ndarray, organism):
    """
    :param organism: Organism to be moved down
    :param world: World GUI
    :param world_state: internal world state
    :return: new position (y, x)
    """
    if organism.position['x'] != world.shape[1] - 1:
        world[organism.position['y'], organism.position['x']] = WORLD_BACKGROUND_COLOR
        world_state[organism.position['y'], organism.position['x']] = 0
        organism.position['x'] += 1
        world[organism.position['y'], organism.position['x']] = organism.color
        world_state[organism.position['y'], organism.position['x']] = organism


def move_left(world: np.ndarray, world_state: np.ndarray, organism):
    """
    :param organism: Organism to be moved down
    :param world: World GUI
    :param world_state: internal world state
    :return: None
    """
    if organism.position['x'] != 0:
        world[organism.position['y'], organism.position['x']] = WORLD_BACKGROUND_COLOR
        world_state[organism.position['y'], organism.position['x']] = 0
        organism.position['x'] -= 1
        world[organism.position['y'], organism.position['x']] = organism.color
        world_state[organism.position['y'], organism.position['x']] = organism


def move_up(world: np.ndarray, world_state: np.ndarray, organism):
    """
    :param organism: Organism to be moved down
    :param world: World GUI
    :param world_state: internal world state
    :return: None
    """
    if organism.position['y'] != world.shape[0] - 1:
        world[organism.position['y'], organism.position['x']] = WORLD_BACKGROUND_COLOR
        world_state[organism.position['y'], organism.position['x']] = 0
        organism.position['y'] += 1
        world[organism.position['y'], organism.position['x']] = organism.color
        world_state[organism.position['y'], organism.position['x']] = organism


def move_down(world: np.ndarray, world_state: np.ndarray, organism):
    """
    :param organism: Organism to be moved down
    :param world: World GUI
    :param world_state: internal world state
    :return: None
    """
    if organism.position['y'] != 0:
        world[organism.position['y'], organism.position['x']] = WORLD_BACKGROUND_COLOR
        world_state[organism.position['y'], organism.position['x']] = 0
        organism.position['y'] -= 1
        world[organism.position['y'], organism.position['x']] = organism.color
        world_state[organism.position['y'], organism.position['x']] = organism
