#for a structured pointcloud (x*y, 1650*1700 or something)
import numpy as np
import pandas as pd
import math
from dataManipulation import raw_to_xyz
from weld_detection_algorithm import subsample_points
import time

# SUBSAMPLE_SIZE = 10 # number of pixels that wil be chosen for the subsampeled Pointcloud, a higher value means a bigger search-grid
GRID_SPACING = 2

PointCloud_size= (1944,1200)
LastChangedPose_point_vector = []

Pointcloud_input:list[list]
points_input:list[list] #chosen points
vectors_input:list[list] #vector representing the direction of the path taken by the end effector (point to point)


#make shure the Poincloud and points are in the same frame

#ALGORITHM:
"""
for each point in points
extract a subset from points based on the location of the current point (this can be done with the indexes rather than the locations of the points)
Get the vector corresponding to the current point and use it to create a plane orthogonaly to the vector (which acts as the normal vector)

In the extracted subset, find all points_ that are on the or close to the plane (either by aprox solution or by creating two planes)

Choose a set distance from the current point and select all points_ on the plane that matches that distance (by some aproximation)
For all remaining points__, choose two points__ that have a distance between each other greater than the distance beteween the current point and the chosen points__

The two chosen points__2 along with the current point are used to find the pose and angle required by the end effector, and a offset is added to make shure the weld is pushed

check the change in angle for all subsequent points, if angle is above threshold, change the pose based on the change in direction

Returns a list containing the position (xyz) and pose (xyzw, or ijkw (quaternions)), with the w part of the quaternbion set to 0
"""

def mean(a,b): 
    return [(a[0]+ b[0])/2, (a[1] +b[1])/2, (a[2] +b[2])/2]

def absolute(a):
    assert len(a) == 3, "not a list/vector/array of length 3"
    return np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)

def is_nan(input:list):
    assert isinstance(input, (list, np.ndarray, tuple)), "The input is not any form of list"
    nans= [np.nan, math.nan, float("nan"), "NaN"]
    for i in input:
        for nan in nans:
            if i==nan or np.isnan(i): #overkill, but can potentially handle more wierd NaN values that we already have
                return True
    return False


def restructure_pointcloud(pointcloud:list[list], pc_shape:tuple[int])-> list[list[list]]:
    # pointcloud = np.array(pointcloud)
    structured_pointcloud = np.reshape(pointcloud, (pc_shape[1], pc_shape[0], 3))
    return structured_pointcloud


# SUBSAMPLE_SIZE = SUBSAMPLE_SIZE//2 # this way the input declares the whole area

def is_points_same(point1:list, point2:list):
    if point1[0] == point2[0] and point1[1] == point2[1] and point1[2] == point2[2]:
        return True
    return False

def get_index_of_point(point:list, Pointcloud:list[list[list]]) -> list:
    """ Takes in a 1D or 2D list of points, returns the index of the point in a list 
    """
    if len(np.shape(Pointcloud)) ==3:
        for i in range(np.shape(Pointcloud)[0]):
            for j in range(np.shape(Pointcloud)[1]):
                if is_points_same(point, Pointcloud[i,j]):
                    return [i,j]
                
    if len(np.shape(Pointcloud)) ==2:
        for i in range(np.shape(Pointcloud)[0]):
            if is_points_same(point, Pointcloud[i]):
                return [i]
    else:
        assert False, "Error, there was no points that could be matched to the input index in get_index_of_point, are you rounding?"
    return []



#UNUSED
def get_3D_min_max(subsampeled_pointcloud:list[list[list]])->list:
    #pick the first instace that is not NaN
    min=[]
    max=[]
    if len(np.shape(subsampeled_pointcloud)) == 3:
        for row in subsampeled_pointcloud:
            for values in row:
                # print(f"{values= }")
                # print(f"{type(values)= }")
                if is_nan(values):#skips instances with NaN-values
                    # print("this is NaN value")
                    continue
                # print("this is not NaN value")
                min= values
                max= values
                break
        for row in subsampeled_pointcloud:
            for values in row:
                if is_nan(values):#skips instances with NaN-values
                    continue
                min = np.minimum(min, values)
                max = np.maximum(max, values)
        return min, max

    elif len(np.shape(subsampeled_pointcloud)) == 2:
        for values in subsampeled_pointcloud:
            # print(f"{values= }")
            # print(f"{type(values)= }")
            if is_nan(values):#skips instances with NaN-values
                # print("this is NaN value")
                continue
            # print("this is not NaN value")
            min= values
            max= values
            break
        for values in subsampeled_pointcloud:
            if is_nan(values):#skips instances with NaN-values
                continue
            min = np.minimum(min, values)
            max = np.maximum(max, values)

    return min, max

def drop_NaN(Pointcloud:list[list]): #this might only work on np.NaN objects, testing needed for the Zivid/Pandas NaN values
    """custom function to drop NaN values, as the whole point needs to be droped, not just the NaN-value
        bad way of doing it as it only checks the first element, but it works for the zivid-output.

    Args:
        pointcloud (list[list]): subsampeled pointcloud
    """
    clean_list = []
    for el in Pointcloud:
        # print(f"{el[0] = }")
        if ~np.isnan(el[0]):
            clean_list.append(el)
    return clean_list







#rotations:
def skew(axis): #returns skew representation of the 3x1 vector
    assert(len(axis) == 3)
    return np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0],0]])

def rotation(axis, theta = 0.0): #Rotation metrix from axis and magnitude w, theta
    return np.eye(3) + np.sin(theta) * skew(axis) + (1 - np.cos(theta)) * skew(axis) @ skew(axis)

def deg_to_rad(deg):
    return (deg/360)*2*np.pi


    
def create_plane2(original_point:list, normal_vector:list, radius:float):
    plot = True
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    plane = []
    #Create a circular mesh grid
    outer_r = radius*1.5

    theta = np.linspace(0, 2 * np.pi, 600)
    r = np.linspace(0, outer_r, 200)
    theta, r = np.meshgrid(theta, r)

    # Convert polar coordinates to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros_like(x)


    x += original_point[0]
    y += original_point[1]
    z += original_point[2]
    z = original_point[2] - (normal_vector[0] * (x- original_point[0]) + normal_vector[1] * (y- original_point[1])) / normal_vector[2]

    x= np.reshape(x, (-1))
    y= np.reshape(y, (-1))
    z= np.reshape(z, (-1))

    # print(f"{np.shape(x)= }")
    # print(f"{np.shape(y)= }")
    # print(f"{np.shape(z)= }")
    # print(x)
    # print(f"{radius = }")
    for i in range(len(x)):
        current_point=[x[i], y[i], z[i]]
        point_dist = np.linalg.norm(np.array(current_point) - np.array(original_point))
        if point_dist <= radius*1.02 and point_dist>= radius*0.98:
            plane.append(current_point)
    
    plane= np.array(plane)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # # ax.plot_surface(x, y, z, color='b', alpha=0.5)

        # # Plot the point on the plane
        ax.scatter(original_point[0], original_point[1], original_point[2], color='r', s=100)

        ax.scatter(plane[:,:1], plane[:,1:2], plane[:,2:], color='b', s=1)

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    return plane


def subsample_pointcloud(pointcloud:list[list[list]], point:list, chosen_point_distance:float):
    import matplotlib.pyplot as plt
    plot = True

    # n=4
    index_point = get_index_of_point(point=point, Pointcloud= pointcloud)
    y0,x0= index_point
    first_box = 140
    assert len(index_point), "no index found for the current point"

    pointcloud = pointcloud[max(0, y0-first_box): min(PointCloud_size[0], y0+first_box), 
                            max(0, x0-first_box): min(PointCloud_size[1], x0+first_box)]

    subsample=[]
    
    # img = np.zeros_like(pointcloud)
    ######################################################
    d = chosen_point_distance
    # print(f"{d= }")

    #find all points within distance d from the original point
    def find_points_within_distance(pointcloud, original_point, distance):
        points_within_distance = []
        for i in range(pointcloud.shape[0]):
            for j in range(pointcloud.shape[1]):
                point = pointcloud[i][j]
                point_dist = np.linalg.norm(point - original_point)
                if point_dist <= distance*1.02 and point_dist>= distance*0.98:
                    points_within_distance.append(point)
        return points_within_distance

    # Find points within the set distance
    subsample = find_points_within_distance(pointcloud, point, d)
    subsample = np.array(subsample)

    if plot:
        x= subsample[:,:1]
        y= subsample[:,1:2]
        z= subsample[:,2:]

        # print(x)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x, y, z, color='b', s=1)
        # ax.plot_surface(x, y, z, color='b', alpha=0.5)

        # Plot the point on the plane

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    return subsample


def choose_first_point(points:list[list],vectors:list[list], dist:float= 50):
    """Get a point used to calculate the first end-effector pose, choosing a point at 'dist' distance into the workpiece from the start.

    Args:
        points (list[list]): The welding-seam-points
        dist (float): The euclidian distance (in mm) into the workpiece where we choose the point

    Returns:
        int: index of the chosen point (and thus corresponding vector)
    """
    index =[]
    start= points[0]

    # for point in points:
    for i in range(len(points)):
        point = points[i]
        vec1 = vectors[i]
        vec1 /= np.linalg.norm(vec1)
        vec2 = point - start
        if np.linalg.norm(vec2) == 0:
            continue
        vec2 /= np.linalg.norm(vec2)
        angle = 2*np.arctan(absolute(absolute(vec2)*vec1 - absolute(vec1)*vec2)/
                           absolute(absolute(vec2)*vec1 + absolute(vec1)*vec2))
        if np.linalg.norm(point- start) >= dist and angle <= np.pi/6: #less than 30 degrees off travel-direction
            index = get_index_of_point(point, points)
            break
    try:
        index = int(index[0])
    except:
        print(f"Could not unpack {index = } to int")
    return index


def get_most_similar_points(grid:list[list], pointcloud:list[list[list]], threshold= 0.5)->list[list]: #realy expensive calculation O(M*N), threshold is not set well yet
    """finds the points with the least euclidian distance from any of the points in the projected plane

    Args:
        grid (list[list]): plane-grid
        pointcloud (list[list[list]]): subsampeled pointcloud
        threshold (float, optional): distance threshold between points, should be about the same distance as between 2-3 points in the pointcloud,
                    Defaults to 0.5, should be calculated or found with testing

    Returns:
        list[list]: _description_
    """
    #flatten pointcloud if its still a 2D list containing the points
    chosen_points = []
    if len(np.shape(pointcloud))==3:
        pointcloud = np.reshape(pointcloud, (-1,3)) #flattens the subsampeled pointcloud as we dont need the shape anymore

    pointcloud = drop_NaN(pointcloud)
    # print("flattened and dropped nans")
    # print(f"{np.shape(grid)= }  ")
    # print(f"{np.shape(pointcloud)= }")
    for pc_point in pointcloud:
        for grid_point in grid:
            if np.linalg.norm(grid_point-pc_point) <= threshold:
                chosen_points.append(pc_point) #important that the poincloud_point is chosen for accuracy/consistency
                # print(f"chose a point!")
            # print(f"{grid_point}")
        # print(f"done big loop iteration for {pc_point = }")
    # print(f"found number of points: {np.shape(chosen_points) = }")
    assert np.shape(chosen_points)[0] >= 2, "Error, to few points were extracted"
    return chosen_points

def get_mapping_points(origin_point:list, points:list[list], input_distance:float):
    assert np.shape(points)[0] >= 2, "Error, to few points were extracted from the previous steps"
    # most_similar= []
    most_similar = points
    selected_points = []
    min_distance = input_distance*0.2 
    # min_distance = 0.5 #arbitrary high value, code farther down can be uncommented to make this a educated guess, but it is not very usefull
    n = 2
    #check points to see the minimum distance between them

    min_angle_rad_90 = 2*3.14
    for point_a in most_similar:
        for point_b in most_similar:
            if is_points_same(point_a, point_b):
                continue
            dist =np.linalg.norm(point_a-point_b)

            vec1 = point_a- origin_point
            vec2 = point_b- origin_point
            angle_rad= 2*np.arctan(absolute(absolute(vec2)*vec1 - absolute(vec1)*vec2)/
                                    absolute(absolute(vec2)*vec1 + absolute(vec1)*vec2)) # new version, complements to William Kahan's formulas

            if dist >= np.sqrt(2*((input_distance*0.9)**2)) and (abs(angle_rad-np.pi/2)< abs(min_angle_rad_90-np.pi/2)): #make shure the points are on different planes
                min_angle_rad_90 = angle_rad
                selected_points = [point_a, point_b]

    assert len(np.shape(selected_points)), "Error, no two points were possible to select satisfying the requirements"
    assert np.shape(selected_points)[0] == 2, f"Error, only one or less points was chosen in the in get_mapping_points np.shape(selected_points): {np.shape(selected_points)}"
    return selected_points



#first iteration: get the pose-vector based on the method above, for the next ones, change its orientation based on the calculated input-vector
def get_pose(point:list, PointCloud:list[list[list]], vector:list, angle_offset:int, chosen_point_distance:float) -> list:
    """calculates the end-effector pose based on input point, vector and pointcloud

    Args:
        point (list): current point for the welding path
        PointCloud (list[list[list]]): input pointcloud (structured and full)
        vector (list): current vector indicating direction of travel
        angle_offset (int): osset added to the pose "away" from direction of travel
        chosen_point_distance (float, optional): distance in mm away from the current point the points used to calculate the pose is chosen by. Defaults to 10.

    Returns:
        list: _description_
    """
    #subsample the pointcloud:


    sub_cloud = subsample_pointcloud(PointCloud, point, chosen_point_distance)
    #get the maximum size of the sampeled area, makes sure all points within this area is taken into account
    plane_grid = create_plane2(point, vector, chosen_point_distance)
    # print("get_most_similar_points()")
    intersection_points= get_most_similar_points(grid=plane_grid, pointcloud=sub_cloud, threshold=0.4)
    # print("get_mapping_points()")
    chosen_points = get_mapping_points(point, intersection_points, chosen_point_distance)
    #calculate the pose


    # current_vec = vectors[0] #get the first vector from the

    middle_pos= mean(chosen_points[0], chosen_points[1]) #works with both array an numpy-array (for numpy this is overkill tho)
    first_pose_vec = middle_pos - point #removed the mean-part as it could potentially cause problems
    first_pose_vec /=absolute(first_pose_vec)


    #rotate the extracted pose around the axis
    vec_axis = vector/absolute(vector)
    # first_pose_vec = rotation(vec_axis, np.pi/2+np.pi/10) @ first_pose_vec
    # first_pose_vec = rotation(vec_axis, np.pi*(1/4 + 1/8)) @ first_pose_vec

    rotation_axis= np.cross(vector, first_pose_vec )
    rotation_axis /= absolute(rotation_axis) #normalize tha axis
    angle_offset_rad = deg_to_rad(angle_offset)

    pose =  rotation(rotation_axis, angle_offset_rad) @ first_pose_vec #NOTE make sure this rotates the correct way (pre/post multiplication and sign of rotation)
    #reversing the pose as it is pointing out of the workpiece
    # pose= -np.array(pose) # add the negative sign if end-effetor is pointing the wrong way, its supposed to be needed 
    pose = pose/absolute(pose)

    # vec_axis = vector/absolute(vector)
    # middle_pos = rotation(vec_axis, np.pi/4 -np.pi) @ middle_pos


    ################################printing and stuff
    # print(f"")
    # print(f"{chosen_points = }")
    # print(f"{point = }")
    vec1 = chosen_points[0]- point
    vec2 = chosen_points[1]- point
    # print(f"{vec1= }")
    # print(f"{vec2= }")
    # print(f"{first_pose_vec= }")
    angle_rad= 2*np.arctan(absolute(absolute(vec2)*vec1 - absolute(vec1)*vec2)/
                           absolute(absolute(vec2)*vec1 + absolute(vec1)*vec2)) # new version, complements to William Kahan's formulas
    angle_deg_90 = 360* angle_rad/(2*np.pi)
    # print(f"{angle_deg_90 = }")
    angle_rad= 2*np.arctan(absolute(absolute(first_pose_vec)*vec1 - absolute(vec1)*first_pose_vec)/
                           absolute(absolute(first_pose_vec)*vec1 + absolute(vec1)*first_pose_vec))
    angle_deg_mid = 360* angle_rad/(2*np.pi)
    # print(f"{angle_deg_mid = }")


    # print("returning pose")
    # print(f"{pose = }")
    # print(f"")

    Plot = True
    #######################
    if Plot:
        import matplotlib.pyplot as plt

        # n=4
        index_point = get_index_of_point(point=point, Pointcloud= PointCloud)
        y0,x0= index_point
        first_box = 140
        assert len(index_point), "no index found for the current point"

        subsample = PointCloud[max(0, y0-first_box): min(PointCloud_size[0], y0+first_box), 
                                max(0, x0-first_box): min(PointCloud_size[1], x0+first_box)]


        subsample = np.reshape(subsample, (-1,3))

        subsample = np.array(subsample)



        # pose_0 = first_pose_vec
        pose_0 = pose


        point0 = point
        size=100
        point1 = [point[0] + pose_0[0]*size, point[1] + pose_0[1]*size, point[2] + pose_0[2]*size]
        x1 = np.linspace(point0[0], point1[0], num= 30)
        y1 = np.linspace(point0[1], point1[1], num= 30)
        z1 = np.linspace(point0[2], point1[2], num= 30)


####    ##########################################################################
        x= subsample[:,:1]
        y= subsample[:,1:2]
        z= subsample[:,2:3]

        print(np.shape(x))
        print(np.shape(y))

        print(np.shape(z))

        # print(x)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x1, y1, z1, color='r', s=50)
        ax.scatter(x, y, z, color='b', s=1, alpha=0.5)



        # ax.plot_surface(x, y, z, color='b', alpha=0.5)

        # Plot the point on the plane

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    return -np.array(pose)
    # return -np.array(first_pose_vec)



def change_pose(pose, current_vec, next_vec):
    """ Changes the selected pose based on the vector for the next points, should not be a big change on a "straight line"

    """
    global LastChangedPose_point_vector

    handle_corner = False ### this is used to handle the corner

    angle = 80
    # angle = 160 #this is set to 160 to "lock" the EE in place and not change direction in the corner, only for test!!!
    if handle_corner:
        angle = 10

    # if len(LastChangedPose_point_vector) != 0:
    #     current_vec= LastChangedPose_point_vector
    #     print(LastChangedPose_point_vector)
 #this is used to change the functionality to only operate on the actual change in the pose-orientation
    if len(LastChangedPose_point_vector)== 0:
        LastChangedPose_point_vector = current_vec
    current_vec = LastChangedPose_point_vector


    #potentially big numerical error due to rounding/normalizing if the vectors are almost alligner or allmost 90 degrees on each other du to division (cross or dot product =0)
    current_vec = np.array(current_vec)
    next_vec = np.array(next_vec)
    pose = np.array(pose)
    assert (absolute(current_vec)* absolute(next_vec)) != 0, f"Error, one of the input-vectors: {current_vec =}, {next_vec = } is a null-vector"

    # Normalizing the input (in case it is not done already)

    current_vec = current_vec/absolute(current_vec)
    next_vec = next_vec/absolute(next_vec)
    # print(f"{current_vec, next_vec =}")
    # print()
    # print(f"{angle_rad = }")
    angle_rad= 2*np.arctan(absolute(absolute(next_vec)*current_vec - absolute(current_vec)*next_vec)/
                           absolute(absolute(next_vec)*current_vec + absolute(current_vec)*next_vec)) # new version, complements to William Kahan's formulas
    # print(f"{angle_rad = }")
    
    angle_deg = 360* angle_rad/(2*np.pi)
    # print(f"{angle_deg = }")
    # axis = np.cross(current_vec, pose)/absolute(np.cross(pose, current_vec)) #normalizing the axis vector

   
    if abs(angle_deg) <= angle: # hvis mindre enn x grader, ikke forandre
        return pose
    else:
        
        axis = np.cross(current_vec, next_vec)/absolute(np.cross(current_vec, next_vec)) #henter akse og normaliserer den
        new_pose=  rotation(axis, angle_rad) @ pose
        new_pose_normalized = new_pose/absolute(new_pose)
        LastChangedPose_point_vector = next_vec
        #maby normalize?
        return new_pose_normalized



def subsample_path_and_estimate_poses(path_df, PointCloud_in:list[list], angle_offset:int= 15, chosen_point_distance:float= 40, starting_point_distance:int= 15):
    poses = []
    """ Calculates the end-effector pose for a welding path based on the surounding walls and end-effectors movement-vector

    Args:
        points (list[list]): The current point corrolating to the current vector.
        vectors (list[list]): The current vector, giving the dirction of the next points, meaning it shows where the end-effector is going to move.
        PointCloud (list[list[list]]): The structured pointcloud, containing x,y,z-coordinates and NaN-values
        angle_offset (int): Given in degrees, gives the angle the end-effctor is going to "push" the weld with, with 0 being perpendicular to the surface, and 90 would
                            send the end effector "straight into" the part it is going to weld, alligning the end-effector with the direction of travel.
        chosen_point_distance (float): distance (in mm?) to where the points used to calualte the pose are chosen

    Returns:
        list: End-effector pose for the input-point
    """
    #extract the first pose
    # NOTE: due to edge-variations we want to extract the 2nd or 3rd point, not the first, then superimpose this on the first (and second/third) point
    poses = []
    subsample_time_start = time.time()
    points, vectors, weld_index= subsample_points(np.array(raw_to_xyz(path_df)))
    susbample_time_end = time.time()
    print(f"subsample path time: {susbample_time_end- subsample_time_start}")


    PointCloud_xyz = PC = raw_to_xyz(PointCloud_in)
    #restructure the pointcloud
    # print(f"{np.shape(PointCloud_in) = }")
    # reshaped_pc = np.reshape(PointCloud_in, (1944, -1))
    # print(f"{np.shape(reshaped_pc)= }")
    # df= pd.DataFrame(reshaped_pc)
    # df.to_csv("Structured_pointcloud", header = None, index = None)
    PointCloud= restructure_pointcloud(PointCloud_xyz, PointCloud_size)
    # print(f"{PointCloud[194:198, 99:102] = }")



    #get the pose
    starting_point_index = choose_first_point(points, vectors, starting_point_distance)
    pose= get_pose(points[starting_point_index], PointCloud, vectors[starting_point_index], angle_offset=angle_offset, chosen_point_distance=chosen_point_distance)
    # plot_pose_context(PointCloud, points[starting_point_index], vectors[starting_point_index], chosen_point_distance,30 )
    # assumes the same pose for the beginning
    for i in range(starting_point_index+1):
        poses.append(pose)

    #for all remaining points, calculate the offset-vector and rotate the pose based on this rotation
    for i in range(np.shape(points)[0]):
        if i <= np.shape(poses)[0]-1: #skip all poses already inserted 2 or 3, ...
            continue
        poses.append(change_pose(poses[i-1], vectors[i-1], vectors[i]))

    if np.shape(poses)[0] !=  np.shape(vectors)[0]:
        print(f"The number of Poses and points/vectors are not the same!!! \n {np.shape(poses)[0], np.shape(vectors)[0] =}")

    poses_time_end = time.time()
    print(f"estimate poses time: {poses_time_end- susbample_time_end}")
    return points, np.array(poses)
    #if there is a logic-error in the previous steps, this should fix it, might be overkill

    # elif np.shape(poses)[0] >  np.shape(vectors)[0]:
    #     while np.shape(poses)[0] >  np.shape(vectors)[0]:
    #         poses.pop(0)
    # elif np.shape(poses)[0] <  np.shape(vectors)[0]:
    #     while np.shape(poses)[0] <  np.shape(vectors)[0]:
    #         vectors.pop(0)





def main():
    df= pd.read_csv("weld_path.csv", header=None)
    PC = raw_to_xyz(pd.read_csv("Front2.csv", header=None))
    # print(f"{PC[1122500:1122520]= }")

    # subsample, vectors, weld_index= subsample_points(np.array(raw_to_xyz(df)))
    # print(f"{np.shape(subsample)= }")
    # print(f"{np.shape(vectors)= }")

#    print(np.array(subsample))
    # print(f"{np.array(vectors)= }")
#    print(f"{len(subsample)= }, {len(vectors) = }")
    points, poses= np.array(subsample_path_and_estimate_poses(df, PC, angle_offset=30, chosen_point_distance=40))
    # print(f"{points = }")

    # print(f"{poses = }")

    return

if __name__ == "__main__":
    main()