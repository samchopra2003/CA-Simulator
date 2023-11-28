import numpy as np

from src.Organism import Organism


def step(world: np.ndarray, world_state: np.ndarray, organisms: list[Organism]):
    """
    :param world: World GUI
    :param world_state: Internal world state
    :param organisms: Organism list
    :return: None
    """
    for org in organisms:
        # TODO: Kill functionality not created yet
        org.step(world, world_state)
