###################################
###################################
# EXAMPLE WITH 5 AGENTS: PART 1
# CREATING THE POSES DATASET
###################################
###################################

###################################
# 1: IMPORTING MODULES
###################################

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

###################################
# 2: CREATING THE POSES
###################################

def pose2y(x):
    x = np.asarray(x)
    return np.where(
        (0 <= x) & (x < 40), 3,  # Constante 3 para 0 <= x < 40
        np.where(
            (40 <= x) & (x < 60), 3 + (x - 40) * (2 - 3) / (60 - 40),  # Lineal decreciente entre 40 y 60
            np.where(
                (60 <= x) & (x <= 100), 2,  # Constante 2 para 60 <= x <= 100
                np.nan  # Fuera del rango
            )
        )
    )

def pose2alpha(x):
    x = np.asarray(x)
    return np.where(
        (0 <= x) & (x < 40), 0,  # Constante 0 para 0 <= x < 40
        np.where(
            (40 <= x) & (x < 60), -np.pi / 4,  # Constante -pi/4 para 40 <= x < 60
            np.where(
                (60 <= x) & (x <= 100), 0,  # Constante 0 para 60 <= x <= 100
                np.nan  # Fuera del rango
            )
        )
    )

poses = np.zeros((100,5, 3))

for t in range(100):
    poses[t][0][0] = t/10
    poses[t][1][0] = t/10
    poses[t][2][0] = t/10
    poses[t][3][0] = t/10
    poses[t][4][0] = t/10
    poses[t][0][1] = 0
    poses[t][1][1] = 1
    poses[t][2][1] = pose2y(t) 
    poses[t][3][1] = 4 
    poses[t][4][1] = 5 
    poses[t][0][2] = 0 
    poses[t][1][2] = 0
    poses[t][2][2] = pose2alpha(t)
    poses[t][3][2] = 0 
    poses[t][4][2] = 0 

np.save('files/poses_5agents.npy', poses)

###################################
# 3: PLOTTING THE TRAJECTORIES
###################################

def draw(i):
    if i < 0 or i >=100:
        i = 0
    circle_radius = 0.2
    fig, ax = plt.subplots(figsize=(5, 3))
    for robot in poses[i]:
        x, y, orientation = robot
        circle = plt.Circle((x, y), circle_radius, color='blue', alpha=0.5)
        ax.add_artist(circle)
        arrow_dx = circle_radius * np.cos(orientation)
        arrow_dy = circle_radius * np.sin(orientation)
        ax.arrow(x, y, arrow_dx, arrow_dy, head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')
    ax.set_title(f'Step {i} of the simulation')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.savefig(f'plots/step_{i}.png')
    plt.close()

frames = []
for i in range(100):
    draw(i)
    frames.append(Image.open(f'plots/step_{i}.png'))
frames[0].save('plots/animation_5agents.gif', format='GIF', save_all=True, append_images=frames[1:], duration=100, loop=0)