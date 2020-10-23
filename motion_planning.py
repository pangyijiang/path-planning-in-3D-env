import numpy as np
from utils_3D import Motion_Planer


if __name__ == "__main__":

    data = [[1, 10,10, 0,20, 20,20, 10,30, 15], [0, 1,1, 4,4, 1,4, 4,1, 17], [0, 18,-4, 24,2, 18,8, 12,2, 17]]
    """
    ****data descripe the buildings or obstacles****
    data: [z, a_x,a_y, b_x,b_y, c_x,c_y, d_x,d_y, h]
        z is botttom of the building in z-anix
        a,b,c,d are vertices of rectangle， a = (a_x, a_y)
        h is the height
    """
    motion_planer = Motion_Planer(data)
    voxmap_start, voxmap_goal = motion_planer.rand_points() #generate a random start point and end point
    
    # Run A* to find a path from start to goal
    paths_r = motion_planer.find_paths(voxmap_start, voxmap_goal, flag_virtual = 1)
    #print("paths_r: ", paths_r)
