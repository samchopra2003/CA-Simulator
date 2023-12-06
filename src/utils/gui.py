import cv2

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
MOVIE_RECORDER_FRAME_RATE = 100


def render_video(img, default_size=True, dsize=None, frame_rate=10, record_movie=False):
    if default_size:
        cv2.imshow('Image', img)
        cv2.waitKey(int(1000 / frame_rate))  # Delay in milliseconds
    else:
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR)
        cv2.imshow('Image', img)
        cv2.waitKey(int(1000 / frame_rate))  # Delay in milliseconds

    global movie_recorder
    if record_movie:
        if movie_recorder is None:
            movie_recorder = MovieRecorder(img, MOVIE_RECORDER_FRAME_RATE)
        else:
            movie_recorder.write(img)
