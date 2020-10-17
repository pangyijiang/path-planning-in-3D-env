import numpy as np
import csv
import random
from utils_3D import Motion_Planer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":

    # Read in obstacle map
    #posX	posY	posZ	halfSizeX	halfSizeY	SizeZ
    #(posX, posY, posZ) is in the middle of the actor
    data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=1)

    motion_planer = Motion_Planer(data)

    # Set goal point
    while True:
        x_goal = random.randint(0, motion_planer.voxmap.shape[0] - 1)
        y_goal = random.randint(0, motion_planer.voxmap.shape[1] - 1)
        z_goal = random.randint(0, motion_planer.voxmap.shape[2] - 1)
        if motion_planer.voxmap[x_goal, y_goal, z_goal] == 0:
            break
    # Voxmap goal
    voxmap_goal = (x_goal, y_goal, z_goal)

    # Set start point
    while True:
        x_start = random.randint(0, motion_planer.voxmap.shape[0] - 1)
        y_start = random.randint(0, motion_planer.voxmap.shape[1] - 1)
        z_start = random.randint(0, motion_planer.voxmap.shape[2] - 1)
        if motion_planer.voxmap[x_start, y_start, z_start] == 0:
            break
    # Voxmap start
    voxmap_start = (x_start, y_start, z_start)
    

    # Run A* to find a path from start to goal
    paths_r, paths_offset = motion_planer.find_paths(voxmap_start, voxmap_goal)
    # print("paths_offset: ", paths_offset)
    # print("Paths_real: ", paths_r)    

    # #virtualize
    # start and end points
    traj_s_e = np.zeros(motion_planer.voxmap.shape, dtype=np.bool)
    traj_s_e[voxmap_start] = True
    traj_s_e[voxmap_goal] = True
    #generated waypoints
    traj = np.zeros(motion_planer.voxmap.shape, dtype=np.bool)
    for path in paths_offset:
        traj[path] = True
    #combine obstacle, sratr-end points, and waypoints
    World = motion_planer.voxmap | traj |traj_s_e
    # set the colors of each object
    colors = np.empty(motion_planer.voxmap.shape, dtype = object)
    colors[motion_planer.voxmap] = 'grey'
    colors[traj] = "red"
    colors[traj_s_e] = "black"
    #plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(World, facecolors=colors, edgecolor='k')
    plt.show()