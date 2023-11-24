import os
import numpy as np
import cv2

from dotenv import load_dotenv


load_dotenv()

# Simulation hyperparameters
WORLD_SIZE_ROWS = int(os.getenv("WORLD_SIZE_ROWS"))
WORLD_SIZE_COLS = int(os.getenv("WORLD_SIZE_COLS"))
STARTING_POPULATION = int(os.getenv("STARTING_POPULATION"))
STEPS_PER_GEN = int(os.getenv("STEPS_PER_GEN"))
GENOME_START_LENGTH = int(os.getenv("GENOME_START_LENGTH"))
HIDDEN_NEURONS = int(os.getenv("HIDDEN_NEURONS"))
MUTATION_RATE_STRUCT = float(os.getenv("MUTATION_RATE_STRUCT"))
MUTATION_RATE_NON_STRUCT = float(os.getenv("MUTATION_RATE_NON_STRUCT"))
FOOD_QUANTITY = int(os.getenv("FOOD_QUANTITY"))
REPRODUCTION_RATE = float(os.getenv("REPRODUCTION_RATE"))
TIME_TO_LIVE = int(os.getenv("TIME_TO_LIVE"))


if __name__ == '__main__':
    world = np.ones((WORLD_SIZE_ROWS, WORLD_SIZE_COLS, 3), dtype=np.uint8) * 255
    world_state = np.zeros((WORLD_SIZE_ROWS, WORLD_SIZE_COLS), dtype=object)

    # Generate starting world population
    for _ in range(STARTING_POPULATION):
        pos = (np.random.randint(0, WORLD_SIZE_ROWS), np.random.randint(0, WORLD_SIZE_COLS))
        color = np.random.randint(0, 256, size=3, dtype=np.uint8)
        world[pos] = color

        # organism data structure










    cv2.imshow('Image', cv2.resize(world, (512, 512), interpolation=cv2.INTER_LINEAR))
    # cv2.imshow('Image', world)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

