
import av
import numpy as np
from PIL import Image
import sys
import numpy
import pygame
from itertools import combinations

numpy.set_printoptions(threshold=sys.maxsize)


VIDEO_PATH = "/home/tesserxt/Downloads/decode_the_drawings_videos/1.mp4"

container = av.open(VIDEO_PATH)
stream = container.streams.video[0]
stream.frames
# Image.fromarray(img).save('hello.jpg')


def get_frame_array():
    for i, frame in enumerate(container.decode(stream)):
        yield np.swapaxes(frame.to_ndarray(format='rgb24'), 1, 0), i


frame_array_generator = get_frame_array()


# where standard deviation between ball area is minimal. [r ~= g ~= ~b]
THRESHOLD = 150

def get_balls_data(frame_arr):
    # Masks - eg: if color_channel > threshold, then that pixel is not background
    r_mask = frame_arr[:, :, 0] > THRESHOLD
    g_mask = (frame_arr[:, :, 1] > THRESHOLD) & ~r_mask
    b_mask = (frame_arr[:, :, 2] > THRESHOLD) & ~r_mask & ~g_mask

    # Get pixel coordinates of ball
    red = np.argwhere(r_mask)
    green = np.argwhere(g_mask & ~r_mask)
    blue = np.argwhere(b_mask & ~r_mask & ~g_mask)

    balls_area = np.array([red.shape[0], green.shape[0], blue.shape[0]])

    radius_red   = int(np.round(np.sqrt(balls_area[0] / np.pi)))
    radius_green = int(np.round(np.sqrt(balls_area[1] / np.pi)))
    radius_blue  = int(np.round(np.sqrt(balls_area[2] / np.pi)))

    # Centers of balls
    center_red   = pygame.math.Vector2(red.mean(axis=0).tolist())
    center_green = pygame.math.Vector2(green.mean(axis=0).tolist())
    center_blue  = pygame.math.Vector2(blue.mean(axis=0).tolist())

    avg_area = int(np.round(balls_area.mean()))
    avg_radius = int(np.round(np.sqrt(avg_area / np.pi)))

    return {
        "red":   [center_red, radius_red],
        "green": [center_green, radius_green],
        "blue":  [center_blue, radius_blue],
    }

