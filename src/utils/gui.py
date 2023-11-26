import cv2


def render_image(img, default_size=True, dsize=None):
    if default_size:
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imshow('Image', cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def render_video(img, default_size=True, dsize=None, frame_rate=10):
    if default_size:
        cv2.imshow('Image', img)
        cv2.waitKey(int(1000 / frame_rate))  # Delay in milliseconds
    else:
        cv2.imshow('Image', cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR))
        cv2.waitKey(int(1000 / frame_rate))  # Delay in milliseconds
