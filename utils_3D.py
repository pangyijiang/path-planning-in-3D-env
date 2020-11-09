from enum import Enum
from queue import PriorityQueue
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as LA
import numpy as np
import random

def r_ints(p1, p2):
    #return the points between the given two points
    I = [i - j for i,j in zip(p2, p1)]
    d = np.sqrt(np.sum(np.square(I)))
    I = [ i/d for i in I]   #unit vector

    p_set = [p1]
    dis = np.sqrt(np.sum(np.square([i - j for i,j in zip(p1, p2)])))
    s = 1
    while dis >= (1.414 + 0.1):
        p_m = [ round(i + j*s) for i,j in zip(p1, I)]
        dis = np.sqrt(np.sum(np.square([i - j for i,j in zip(p2, p_m)])))
        s = s + 1
        if p_m not in p_set:
            p_set.append(p_m)

    if p2 not in p_set:
        p_set.append(p2)
    return p_set

class Motion_Planer:
    def __init__(self, data):
        """
        ****data descripe the buildings or obstacles****
        data: [z, a_x,a_y, b_x,b_y, c_x,c_y, d_x,d_y, h]
            z is botttom of the building in z-anix
            a,b,c,d are vertices of rectangle， a = (a_x, a_y)
            h is the height
        """
        self.voxmap, self.offset = create_voxmap(data)
        #print("map_offset:", self.offset)
    def find_paths(self, voxmap_start, voxmap_goal, simplify = 1, flag_offset = 1, flag_virtual = 0):
        if flag_offset == 1:
            voxmap_start = [round(i - j) for i,j in zip(voxmap_start, self.offset)]
            voxmap_goal = [round(i - j) for i,j in zip(voxmap_goal, self.offset)]
        paths, _ = a_star_3D(self.voxmap, heuristic, tuple(voxmap_start), tuple(voxmap_goal))
        if simplify == 1:
            num_key_points = [0]
            num = len(paths)
            i = 0
            point_s = paths[0]
            point_e = paths[1]
            while i <= (num-1):
                points_m = r_ints(point_s, point_e)
                obs_r = []
                for point in points_m:
                    obs_r.append(self.voxmap[point[0], point[1], point[2]])
                if True in obs_r:
                    num_key_points.append(i)
                    point_s = paths[i]
                    point_e = paths[i+1]
                    continue
                i = i + 1
                try:
                    point_e = paths[i]
                except:
                    pass
            num_key_points.append(num - 1)  #add the last point
            paths_key = [ paths[i] for i in num_key_points]
            paths_key_r = [ [i[0]+self.offset[0], i[1]+self.offset[1], i[2]+self.offset[2]] for i in paths_key[1:]]
        paths_r = [ [i[0]+self.offset[0], i[1]+self.offset[1], i[2]+self.offset[2]] for i in paths[1:]]
        #print("paths: ", paths)
        #print("paths_r: ", paths_r)
        if flag_virtual == 1:
            # #virtualize
            # start and end points
            traj_s_e = np.zeros(self.voxmap.shape, dtype=np.bool)
            traj_s_e[voxmap_start[0]][voxmap_start[1]][voxmap_start[2]] = True
            traj_s_e[voxmap_goal[0]][voxmap_goal[1]][voxmap_goal[2]] = True
            traj = np.zeros(self.voxmap.shape, dtype=np.bool)
            for path in paths[1:-1]:
                traj[path] = True
            traj_key = np.zeros(self.voxmap.shape, dtype=np.bool)
            for key in paths_key:
                traj_key[key] = True
            #combine obstacle, sratr-end points, and waypoints
            World = self.voxmap |traj_s_e |traj
            # set the colors of each object
            colors = np.empty(self.voxmap.shape, dtype = object)
            colors[self.voxmap] = 'grey'
            colors[traj_s_e] = "black"
            colors[traj] = "red"
            colors[traj_key] = "blue"
            print("Prepare to show the Virtual Worlf of motion planning")
            #plot
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.voxels(World, facecolors=colors, edgecolor='k')
            plt.show()
        if simplify == 1:
            return paths_key_r
        else:
            return paths_r
    def rand_points(self):
        while True:
            n_goal = random.randint(0, self.voxmap.shape[0] - 1)
            e_goal = random.randint(0, self.voxmap.shape[1] - 1)
            alt_goal = random.randint(0, self.voxmap.shape[2] - 1)
            if self.voxmap[n_goal, e_goal, alt_goal] == 0:
                break
        # Voxmap goal
        voxmap_goal = [n_goal, e_goal, alt_goal]
        while True:
            n_start = random.randint(0, self.voxmap.shape[0] - 1)
            e_start = random.randint(0, self.voxmap.shape[1] - 1)
            alt_start = random.randint(0, self.voxmap.shape[2] - 1)
            if self.voxmap[n_start, e_start, alt_start] == 0:
                break
        # Voxmap start
        voxmap_start = [n_start, e_start, alt_start]
        voxmap_start = [round(i + j) for i,j in zip(voxmap_start, self.offset)]
        voxmap_goal = [round(i + j) for i,j in zip(voxmap_goal, self.offset)]
        return voxmap_goal, voxmap_start

def heuristic(position, goal_position):
    return LA.norm(np.array(position) - np.array(goal_position))

def create_voxmap(data, map_margin = 5, safety_distance = 0, voxel_size = 1):
    #data: (z, a, b, c, d, h), a,b,c,d are vertices of rectangle. h is the height.
    """
    Returns a grid representation of a 3D configuration space
    based on given obstacle data and safety distance

    The 'target_altitude': highest point + target_altitude is the height of the map.
    The 'voxel_size' argument sets the resolution of the voxel map.
    """
    # minimum and maximum coordinates
    data = np.array(data)
    # temp = data[:, 1][0]# + data[:, 2][0] + data[:, 3][0] + data[:, 4][0]
    x_all = np.hstack((data[:, 1], data[:, 3], data[:, 5], data[:, 7]))
    y_all = np.hstack((data[:, 2], data[:, 4], data[:, 6], data[:, 8]))
    #start point
    s_p_x = np.floor(np.min(x_all))  
    s_p_y = np.floor(np.min(y_all))  
    x_offset = s_p_x - map_margin
    y_offset = s_p_y - map_margin
    #end point
    e_p_x = np.ceil(np.max(x_all))  
    e_p_y = np.ceil(np.max(y_all))  
    # z-axis
    z_offset = np.min(data[:, 0])

    x_size = int(e_p_x - s_p_x + map_margin*2)
    y_size = int(e_p_y - s_p_y + map_margin*2)
    z_size = int(np.max(data[:, 0]- z_offset + data[:, 9]) + map_margin)

    voxmap = np.zeros((x_size, y_size, z_size), dtype=np.bool)

    for i in range(data.shape[0]):
        z, a_x,a_y, b_x,b_y, c_x,c_y, d_x,d_y, h = data[i, :]
        a = [a_x,a_y]; b = [b_x,b_y]; c = [c_x,c_y]; d = [d_x,d_y]
        x_index, y_index = map_color_seg(a, b, c, d)
        x_index = x_index - x_offset
        y_index = y_index - y_offset
        z = int(z - z_offset)
        voxmap[x_index.astype(int), y_index.astype(int), z:(z+h)] = True

    offset = [x_offset, y_offset, z_offset]
    return voxmap, offset

def map_color_seg(a,b,c,d):
    def Sort4Points(points_4):
        point_c = [ (a+b+c+d)/4 for a,b,c,d in zip(points_4[0], points_4[1], points_4[2], points_4[3])]
        dic_a_p = {}
        for point in points_4:
            angle = np.arctan2(point[1]-point_c[1], point[0]-point_c[0])
            dic_a_p[angle] = point
        return [dic_a_p[k] for k in sorted(dic_a_p.keys())]
    def linear_k_b(point1, point2):
        if point1[0] == point2[0]:
            k = 0
        else:
            k = (point1[1] - point2[1])/(point1[0] - point2[0])
        b = point1[1] - k*point1[0]
        return k, b

    # the vertices should be in order
    a,b,c,d = Sort4Points([a,b,c,d])
    ab_linear = linear_k_b(a, b)
    bc_linear = linear_k_b(b, c)
    cd_linear = linear_k_b(c, d)
    da_linear = linear_k_b(d, a)

    #start point
    s_p_x = np.floor(np.min([a[0], b[0], c[0], d[0]]))  
    s_p_y = np.floor(np.min([a[1], b[1], c[1], d[1]]))  
    #end point
    e_p_x = np.ceil(np.max([a[0], b[0], c[0], d[0]]))  
    e_p_y = np.ceil(np.max([a[1], b[1], c[1], d[1]]))  

    offset = [s_p_x, s_p_y]
    size = [int(e_p_x - s_p_x), int(e_p_y - s_p_y)]

    ab_map = np.zeros((size[0], size[1]))
    bc_map = np.zeros((size[0], size[1]))
    cd_map = np.zeros((size[0], size[1]))
    da_map = np.zeros((size[0], size[1]))

    for x in range(size[0]):
        for y in range(size[1]):
            #two color for ab_map
            if (x+offset[0])*ab_linear[0] + ab_linear[1] - (y+offset[1]) >=0:
                ab_map[x][y] = 7
            else:
                ab_map[x][y] = 3
            #two color for cd_map
            if (x+offset[0])*cd_linear[0] + cd_linear[1] - (y+offset[1]) <=0:
                cd_map[x][y] = -7
            else:
                cd_map[x][y] = -3

            #two color for bc_map
            if (x+offset[0])*bc_linear[0] + bc_linear[1] - (y+offset[1]) >=0:
                bc_map[x][y] = 19
            else:
                bc_map[x][y] = 37
            #two color for da_map
            if (x+offset[0])*da_linear[0] + da_linear[1] - (y+offset[1]) <=0:
                da_map[x][y] = -19
            else:
                da_map[x][y] = -37
    map_all = ab_map + bc_map + cd_map + da_map
    map_all[np.nonzero(map_all)]=1
    [x_index, y_index] = np.where(map_all == 0)
    x_index = x_index + offset[0]
    y_index = y_index + offset[1]
    return x_index.astype(int), y_index.astype(int)

### 3D A*
class Action3D(Enum):
    """
    An action is represented by a 4 element tuple.

    The first 3 values are the delta of the action relative
    to the current voxel position. The final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 0, 1)
    EAST = (0, 1, 0, 1)
    NORTH = (-1, 0, 0, 1)
    SOUTH = (1, 0, 0, 1)

    # Diagonal motions
    N_WEST = (-1, -1, 0, np.sqrt(2))
    N_EAST = (-1, 1, 0, np.sqrt(2))
    S_WEST = (1, -1, 0, np.sqrt(2))
    S_EAST = (1, 1, 0, np.sqrt(2))

    # Up & down motions
    UP = (0, 0, 1, 1)
    DOWN = (0, 0, -1, 1)

    @property
    def cost(self):
        return self.value[3]

    @property
    def delta(self):
        return (self.value[0], self.value[1], self.value[2])

def valid_actions_3D(voxel, current_node):
    """
    Returns a list of valid actions given a voxel and current node.
    """
    valid_actions = list(Action3D)
    n, m, max_alt = voxel.shape[0] - 1, voxel.shape[1] - 1, voxel.shape[2] - 1
    x, y, z = current_node

    # check if the node is off the voxel or
    # it's an obstacle
    if z - 1 < 0 or voxel[x, y, z - 1] == 1:
        valid_actions.remove(Action3D.DOWN)
    if z + 1 > max_alt or voxel[x, y, z + 1] == 1:
        valid_actions.remove(Action3D.UP)

    if x - 1 < 0 or voxel[x - 1, y, z] == 1:
        valid_actions.remove(Action3D.NORTH)
        valid_actions.remove(Action3D.N_WEST)
        valid_actions.remove(Action3D.N_EAST)
    if x + 1 > n or voxel[x + 1, y, z] == 1:
        valid_actions.remove(Action3D.SOUTH)
        valid_actions.remove(Action3D.S_WEST)
        valid_actions.remove(Action3D.S_EAST)

    if y - 1 < 0 or voxel[x, y - 1, z] == 1:
        valid_actions.remove(Action3D.WEST)
        if Action3D.N_WEST in valid_actions:
            valid_actions.remove(Action3D.N_WEST)
        if Action3D.S_WEST in valid_actions:
            valid_actions.remove(Action3D.S_WEST)
    if y + 1 > m or voxel[x, y + 1, z] == 1:
        valid_actions.remove(Action3D.EAST)
        if Action3D.N_EAST in valid_actions:
            valid_actions.remove(Action3D.N_EAST)
        if Action3D.S_EAST in valid_actions:
            valid_actions.remove(Action3D.S_EAST)

    return valid_actions

def a_star_3D(voxel, h, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if np.all(current_node == start):
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        if np.all(current_node == goal):
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions_3D(voxel, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1], current_node[2] + da[2])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))

    if found:
        # retrace steps
        n = goal

        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost
