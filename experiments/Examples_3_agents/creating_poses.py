###################################
###################################
# EXAMPLE WITH 3 AGENTS: PART 1
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

def pose0x(x):
    return x/10

def pose1x(x):
    x = np.asarray(x)
    return np.where(
        (0 <= x) & (x < 45), 2 + (x - 0) * (6.5 - 2) / (45 - 0),
        np.where(
            (45 <= x) & (x <= 55), 6.5,
            np.where(
                (55 < x) & (x <= 100), 6.5 + (x - 55) * (11 - 6.5) / (100 - 55),
                np.nan
            )
        )
    )

def pose2x(x):
   return x/10 + 3

poses = np.zeros((100,3, 3))

for t in range(100):
    poses[t][0][0] = pose0x(t) 
    poses[t][1][0] = pose1x(t)
    poses[t][2][0] = pose2x(t) 
    poses[t][0][1] = 0
    poses[t][1][1] = 0
    poses[t][2][1] = 0 
    poses[t][0][2] = 0 
    poses[t][1][2] = 0
    poses[t][2][2] = 0 

np.save('files/poses_3agents.npy', poses)

###################################
# 3: PLOTTING THE TRAJECTORIES
###################################

def draw(i):
    if i < 0 or i >=100:
        i = 0
    circle_radius = 0.2
    fig, ax = plt.subplots(figsize=(8, 1.5))
    for robot in poses[i]:
        x, y, orientation = robot
        circle = plt.Circle((x, y), circle_radius, color='blue', alpha=0.5)
        ax.add_artist(circle)
        arrow_dx = circle_radius * np.cos(orientation)
        arrow_dy = circle_radius * np.sin(orientation)
        ax.arrow(x, y, arrow_dx, arrow_dy, head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 0.5)
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
frames[0].save('plots/animation_3agents.gif', format='GIF', save_all=True, append_images=frames[1:], duration=100, loop=0)