import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()

WORLD_BACKGROUND_COLOR = int(os.getenv("WORLD_BACKGROUND_COLOR"))
WORLD_SIZE_ROWS = int(os.getenv("WORLD_SIZE_ROWS"))
WORLD_SIZE_COLS = int(os.getenv("WORLD_SIZE_COLS"))

KILL_RADIUS = int(os.getenv("KILL_RADIUS"))
KILL_PROBABILITY = float(os.getenv("KILL_PROBABILITY"))


def move_right(world: np.ndarray, world_state: np.ndarray, organism):
    """
    :param organism: Organism to be moved right
    :param world: World GUI
    :param world_state: internal world state
    :return: new position (y, x)
    """
    if organism.position['x'] < WORLD_SIZE_COLS - 1 and \
            world_state[organism.position['y'], organism.position['x'] + 1] == 0:
        world[organism.position['y'], organism.position['x']] = WORLD_BACKGROUND_COLOR
        world_state[organism.position['y'], organism.position['x']] = 0
        organism.position['x'] += 1
        world[organism.position['y'], organism.position['x']] = organism.color
        world_state[organism.position['y'], organism.position['x']] = organism
        organism.last_move = 0


def move_left(world: np.ndarray, world_state: np.ndarray, organism):
    """
    :param organism: Organism to be moved left
    :param world: World GUI
    :param world_state: internal world state
    :return: None
    """
    if organism.position['x'] > 0 and world_state[organism.position['y'], organism.position['x'] - 1] == 0:
        world[organism.position['y'], organism.position['x']] = WORLD_BACKGROUND_COLOR
        world_state[organism.position['y'], organism.position['x']] = 0
        organism.position['x'] -= 1
        world[organism.position['y'], organism.position['x']] = organism.color
        world_state[organism.position['y'], organism.position['x']] = organism
        organism.last_move = 1


def move_up(world: np.ndarray, world_state: np.ndarray, organism):
    """
    :param organism: Organism to be moved up
    :param world: World GUI
    :param world_state: internal world state
    :return: None
    """
    if organism.position['y'] > 0 and \
            world_state[organism.position['y'] - 1, organism.position['x']] == 0:
        world[organism.position['y'], organism.position['x']] = WORLD_BACKGROUND_COLOR
        world_state[organism.position['y'], organism.position['x']] = 0
        organism.position['y'] -= 1
        world[organism.position['y'], organism.position['x']] = organism.color
        world_state[organism.position['y'], organism.position['x']] = organism
        organism.last_move = 2


def move_down(world: np.ndarray, world_state: np.ndarray, organism):
    """
    :param organism: Organism to be moved down
    :param world: World GUI
    :param world_state: internal world state
    :return: None
    """
    if organism.position['y'] < WORLD_SIZE_ROWS - 1 and \
            world_state[organism.position['y'] + 1, organism.position['x']] == 0:
        world[organism.position['y'], organism.position['x']] = WORLD_BACKGROUND_COLOR
        world_state[organism.position['y'], organism.position['x']] = 0
        organism.position['y'] += 1
        world[organism.position['y'], organism.position['x']] = organism.color
        world_state[organism.position['y'], organism.position['x']] = organism
        organism.last_move = 3


def die(world: np.ndarray, world_state: np.ndarray, organism):
    world[organism.position['y'], organism.position['x']] = WORLD_BACKGROUND_COLOR
    world_state[organism.position['y'], organism.position['x']] = 0
    organism.alive = False


def kill(world: np.ndarray, world_state: np.ndarray, organism):
    """
    Kill in forward direction
    :param world: World GUI
    :param world_state: internal world state
    :param organism: Organism to be moved down
    :return:
    """
    # TODO: DONT KILL ORGANISMS IN SAME SPECIES
    forward_pos_x = organism.position['x']
    forward_pos_y = organism.position['y']
    if organism.last_move == 0:  # right
        forward_pos_x += 1
    elif organism.last_move == 1:  # left
        forward_pos_x -= 1
    if organism.last_move == 2:  # up
        forward_pos_y -= 1
    elif organism.last_move == 3:  # down
        forward_pos_y += 1

    if WORLD_SIZE_COLS - 1 >= forward_pos_x >= 0 and WORLD_SIZE_ROWS - 1 >= forward_pos_y >= 0:
        org = world_state[forward_pos_y, forward_pos_x]
        if org != 0:
            # TODO: KILL PROBABILITY SHOULD INCREASE WITH HIGHER GAP IN FITNESS
            if np.random.random() < KILL_PROBABILITY and organism.fitness > org.fitness and \
                    organism.species != org.species:
                org.die(world, world_state)
                organism.kills += 1
