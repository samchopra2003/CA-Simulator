import os

import numpy as np
from dotenv import load_dotenv

from Organism import Organism
from utils.gui import render_video
from utils.step import step, Monitor, segregate_species

load_dotenv()

# Simulation hyperparameters
WORLD_SIZE_ROWS = int(os.getenv("WORLD_SIZE_ROWS"))
WORLD_SIZE_COLS = int(os.getenv("WORLD_SIZE_COLS"))
WORLD_BACKGROUND_COLOR = int(os.getenv("WORLD_BACKGROUND_COLOR"))
STARTING_POPULATION = int(os.getenv("STARTING_POPULATION"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS"))
STEPS_PER_GEN = int(os.getenv("STEPS_PER_GEN"))
GENOME_START_LENGTH = int(os.getenv("GENOME_START_LENGTH"))
HIDDEN_NEURONS = int(os.getenv("HIDDEN_NEURONS"))
MUTATION_RATE_STRUCT = float(os.getenv("MUTATION_RATE_STRUCT"))
MUTATION_RATE_NON_STRUCT = float(os.getenv("MUTATION_RATE_NON_STRUCT"))
FOOD_QUANTITY = int(os.getenv("FOOD_QUANTITY"))
TIME_TO_LIVE = int(os.getenv("TIME_TO_LIVE"))
VIDEO_RENDER_FREQ = int(os.getenv("VIDEO_RENDER_FREQ"))
MONITOR_FREQ = int(os.getenv("MONITOR_FREQ"))

if __name__ == '__main__':
    world = np.ones((WORLD_SIZE_ROWS, WORLD_SIZE_COLS, 3), dtype=np.uint8) * WORLD_BACKGROUND_COLOR
    world_state = np.zeros((WORLD_SIZE_ROWS, WORLD_SIZE_COLS), dtype=object)
    monitor = Monitor()

    # Generate starting population
    organism_list = []
    for _ in range(STARTING_POPULATION):
        pos = (np.random.randint(0, WORLD_SIZE_ROWS), np.random.randint(0, WORLD_SIZE_COLS))
        if world_state[pos] == 0:
            color = np.random.randint(0, 256, size=3, dtype=np.uint8)
            world[pos] = color
            # create organism
            organism = Organism(pos, color)
            organism_list.append(organism)
            world_state[pos] = organism

    monitor.all_time_population = len(organism_list)

    # compute distinct species
    species = {}
    segregate_species(organism_list, species)
    monitor.num_species = len(species)
    monitor.species_names = [name for name, _ in species.items()]
    monitor.species_populations = [len(orgs) for _, orgs in species.items()]
    monitor.species_fitnesses = [0 for _ in species]

    monitor.species_colors = {name: np.random.randint(0, 256, size=3, dtype=np.uint8)
                              for name, _ in species.items()}

    # Start simulation
    for gen in range(NUM_GENERATIONS):
        for gen_step in range(STEPS_PER_GEN):
            step(world, world_state, organism_list, species, species_specific_colors=True)
            monitor.all_time_population = max(monitor.all_time_population, monitor.total_population)

            if (gen_step + 1) % VIDEO_RENDER_FREQ == 0:
                render_video(world.copy(), monitor,
                             default_size=False, dsize=(512, 512), frame_rate=100, record_movie=True,
                             species_specific_colors=True)

            monitor.generation = gen
            monitor.gen_step = gen_step
            if (gen_step + 1) % MONITOR_FREQ == 0:
                monitor.print_monitor()
