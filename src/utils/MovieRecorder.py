import cv2
import os


class MovieRecorder:
    def __init__(self, img, frame_rate):
        directory_path = '../assets/movies'
        base_file_name = 'movie'
        file_extension = '.mp4'

        existing_files = [f for f in os.listdir(directory_path) if f.endswith(file_extension)]
        if existing_files:
            existing_files.sort()
            last_file_number = int(existing_files[-1].replace(base_file_name, '').replace(file_extension, ''))
            new_file_name = f"{base_file_name}{last_file_number + 1}{file_extension}"
        else:
            new_file_name = f"{base_file_name}0{file_extension}"

        movie_file = os.path.join(directory_path, new_file_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.movie = cv2.VideoWriter(movie_file, fourcc, frame_rate, (img.shape[1], img.shape[0]))
        self.movie.write(img)

    def write(self, img):
        self.movie.write(img)
