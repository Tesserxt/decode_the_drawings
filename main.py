
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



pygame.init()
display = pygame.display.set_mode((1280, 720))
trail_surf = pygame.Surface((1280, 720), pygame.SRCALPHA)
clock = pygame.time.Clock()
running = True
triangle_points = [] # x, y coordinates of triangle - center of each ball
font = pygame.font.SysFont("Arial", 16)


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    #surf = pygame.surfarray.make_surface(frame_arr)

    # display.fill((0, 0, 0))
    #display.blit(surf, (0, 0))


    frame_arr, i = next(frame_array_generator)
    if i > stream.frames - 2:
        pygame.quit()
        break
    balls_data = get_balls_data(frame_arr)


    display.fill((0,0,0))


    for key in balls_data:
        c, r = balls_data[key]
        pygame.draw.circle(display, center=c, radius=r, color=pygame.Color(key), width=0)
        pygame.draw.circle(display, center=c, radius=108, color=pygame.Color(key), width=1)

        triangle_points.append(c)

    eul = []
    for key1, key2 in combinations(balls_data.keys(), 2):
        c1 = balls_data[key1][0] #center of first ball
        c2 = balls_data[key2][0] #center of second ball
        eucli_distance = c1.distance_to(c2)
        eul.append(eucli_distance)

        pygame.draw.line(display, color=pygame.Color('yellow'), start_pos=c1, end_pos=c2, width=1)
        text = font.render(f'{int(eucli_distance)}', True, pygame.Color('cyan'))  # Text, Anti-aliasing, Color
        display.blit(text, ((c1 + c2)/2))


    centroid = np.mean(triangle_points, axis=0) #center of triangle


    a = 2 * int(np.mean(eul))



    pygame.gfxdraw.pixel(trail_surf, int(centroid[0]), int(a), pygame.Color('white'))

    display.blit(trail_surf, (0,0))

    pygame.display.update()
    pygame.display.flip()
    clock.tick(120)

pygame.quit()



