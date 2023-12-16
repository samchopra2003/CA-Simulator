import cv2
import numpy as np

from src.utils.MovieRecorder import MovieRecorder


def render_image(img, default_size=True, dsize=None):
    if default_size:
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imshow('Image', cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


movie_recorder = None
MOVIE_RECORDER_FRAME_RATE = 20


def render_video(img, monitor, default_size=True, dsize=None, frame_rate=10, record_movie=False,
                 species_specific_colors=False):
    if default_size:
        # TODO: Add monitor support for default size
        cv2.imshow('Image', img)
        cv2.waitKey(int(1000 / frame_rate))  # Delay in milliseconds
    else:
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR)

        # add Monitor data
        height, width, channels = img.shape
        extra_columns = 200
        new_width = width + extra_columns
        new_image = np.zeros((height, new_width, channels), dtype=np.uint8)
        new_image[:, extra_columns:] = img

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_color = (255, 255, 255)
        thickness = 1

        position = (5, 10)
        cv2.putText(new_image, f"Generation {monitor.generation}, Step {monitor.gen_step}",
                    position, font, font_scale, font_color, thickness)
        position = (5, 20)
        cv2.putText(new_image, "-------------",
                    position, font, font_scale, font_color, thickness)
        position = (5, 30)
        cv2.putText(new_image, f"All-time population: {monitor.all_time_population}",
                    position, font, font_scale, font_color, thickness)
        position = (5, 40)
        cv2.putText(new_image, f"Current population: {monitor.total_population}",
                    position, font, font_scale, font_color, thickness)
        position = (5, 50)
        cv2.putText(new_image, f"Number of males: {monitor.num_males}",
                    position, font, font_scale, font_color, thickness)
        position = (5, 60)
        cv2.putText(new_image, f"Number of females: {monitor.num_females}",
                    position, font, font_scale, font_color, thickness)
        position = (5, 70)
        cv2.putText(new_image, f"Reproductions: {monitor.num_reproductions}",
                    position, font, font_scale, font_color, thickness)
        position = (5, 80)
        cv2.putText(new_image, f"Avg. fitness: {monitor.avg_fitness:.3f}",
                    position, font, font_scale, font_color, thickness)
        position = (5, 90)
        cv2.putText(new_image, f"Total number of mutations: {monitor.total_mutations}",
                    position, font, font_scale, font_color, thickness)
        position = (5, 100)
        cv2.putText(new_image, f"Total number of predators: {monitor.num_predators}",
                    position, font, font_scale, font_color, thickness)
        position = (5, 110)
        cv2.putText(new_image, f"Organisms killed: {monitor.num_killed}",
                    position, font, font_scale, font_color, thickness)

        position = (5, 150)
        cv2.putText(new_image, f"Number of species: {monitor.num_species}",
                    position, font, font_scale, font_color, thickness)
        y = 150
        for idx, name in enumerate(monitor.species_names):
            position = (5, y + (idx + 1) * 15)
            cv2.putText(new_image, f"{idx+1}. {name}",
                        position, font, font_scale, font_color, thickness)

            if species_specific_colors:
                top_left = (120, y + (idx + 1) * 15 - 10)
                bottom_right = (130, y + (idx + 1) * 15)
                color_values = monitor.species_colors[name]
                color = tuple(map(int, color_values))
                cv2.rectangle(new_image, top_left, bottom_right, color, thickness=cv2.FILLED)

            position = (5, y + (idx + 2) * 15)
            cv2.putText(new_image, f"Pop: {monitor.species_populations[idx]}, "
                                   f"Fitness: {monitor.species_fitnesses[idx]:.3f}",
                        position, font, font_scale, font_color, thickness)
            y += 15

            if idx == 11:   # max number of species that will fit on screen
                break

        img = new_image

        cv2.imshow('Image', img)
        cv2.waitKey(int(1000 / frame_rate))  # Delay in milliseconds

    global movie_recorder
    if record_movie:
        if movie_recorder is None:
            movie_recorder = MovieRecorder(img, MOVIE_RECORDER_FRAME_RATE)
        else:
            movie_recorder.write(img)
