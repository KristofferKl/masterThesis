from tkinter import *
from PIL import Image, ImageTk
import skimage as ski
import numpy as np
import pandas as pd
import time

from dataManipulation import *
from end_effector_pose_sorted_PC import subsample_path_and_estimate_poses
"""
last updated: 14th june, 2024
This is the main code to run for the project
When the window pops up (left)click on 3 (or more) points that you want to create a weld-line between. 
These clicks also create the templates.
To continiue press enter.
To exit the window before completing the clicks, either press "q" or close it manually.
The extracted lines are then converted to a subsampeled weld-path
The prosess then saves the resulting weld path to "OUTPU.csv" and "OUTPUT_OROGINAL.csv" 

By setting "runde2" to True, you activate the template-matching-part, which uses the previous template and weld path, 
and translates it based on the offset between the prevois part and the new one 

To change what portion of the weld-line to choose, change the part used in the do_subsample_extract_transform() call to either...._comb or ....1


"""


#start of functions
def find_nearest_white(img, target):
    target = [target[1], target[0]]
    nonzero = np.argwhere(img == 255)
    distances = np.sqrt((nonzero[:,0] - target[0]) ** 2 + (nonzero[:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)
    coordinate = nonzero[nearest_index]
    return [coordinate[1], coordinate[0]]


#####
def get_traversed_image(coords, image_path:str = 'skeleton.png', save:bool=True):
    image = ski.io.imread(image_path, "gray")
    # Define directions for traversing (8-connectivity)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    # Initialize a visited matrix
    visited = np.zeros_like(image, dtype=np.uint8)

    def traverse(start:list[int], end:list[int], visited): # this is bugged, currently only works correctly for points from top left towards bottom right, not the other way
        start = (start[0], start[1])
        end = (end[0], end[1])
        stack = [start]
        while stack:
            current = stack.pop()
            if current == end:
                return True
            visited[current[1], current[0]] = 1
            for direction in directions:
                next_pixel = (current[0] + direction[1], current[1] + direction[0])
                if (0 <= next_pixel[0] < image.shape[1] and
                        0 <= next_pixel[1] < image.shape[0] and
                        image[next_pixel[1], next_pixel[0]] == 255 and
                        not visited[next_pixel[1], next_pixel[0]]):
                    stack.append(next_pixel)
        return False
    
    for i in range(len(coords)-1):
        traverse(coords[i], coords[i+1], visited)
    result_image = np.zeros_like(image)
    result_image[np.where(visited == 1)] = 255
    # Save the resulting image
    if save:
        ski.io.imsave("traversed.png", result_image)
    return result_image
#####

def get_actual_cordinates(image,coordinates ):
    actual_coordinates = []
    actual_coordinates.append([find_nearest_white(image, coordinates[0])[0], find_nearest_white(image, coordinates[0])[1]])
    for i in range(len(coordinates)):
        if len(coordinates)-1 > i:
            actual_coordinates.append([find_nearest_white(image, coordinates[i+1])[0],find_nearest_white(image, coordinates[i+1])[1]]) 
    return actual_coordinates

def draw_lines(image, coordinates):
    for i in range(len(coordinates)-1):
        rr, cc = ski.draw.line(r0=coordinates[i][1], c0=coordinates[i][0], 
                               r1=coordinates[i+1][1], c1=coordinates[i+1][0])
        image[rr,cc] = 255
        # print(rr,cc)
    # ski.io.imsave("weld_lines.png", image, check_contrast=False)
    return image



def make_poses_transform_comapatible(poses_df):
    assert poses_df.shape[1] == 3, "The poses are in the wrong form, each pose is not of length 3"
    padding = pd.Series(np.zeros(poses_df.shape[0]))
    return pd.concat([poses_df, padding], axis=1)


def do_subsample_extract_transform(extracted_df, pointcloud, robot_matrix, camera_matrix, angle_offset=15, chosen_point_distance=10,starting_point_distance = 15 , pose_as_quaternion_xyzw:bool = True):

    points, poses = subsample_path_and_estimate_poses(extracted_df, pointcloud, angle_offset, chosen_point_distance, starting_point_distance= starting_point_distance)
    subsample_and_poses_time_end = time.time()
    points_df = pd.DataFrame(points)
    transformed_points_df = df_transformation(matrix_rob=robot_matrix, matrix_cam= camera_matrix, points= points_df)

    poses_df = pd.DataFrame(poses)
    poses_df_comp = make_poses_transform_comapatible(poses_df) #important to add zero-padding to make the transformation act as a pure rotation on the vector, df_transformation adds ones as padding if the length of each input point is 3 instead of 4
    transformed_poses_df = df_transformation(matrix_rob=robot_matrix, matrix_cam= camera_matrix, points= poses_df_comp)

    transformed_points_df = raw_to_xyz(transformed_points_df)
    transformed_points = np.array(transformed_points_df)

    if not pose_as_quaternion_xyzw:
        transformed_poses_df = raw_to_xyz(transformed_poses_df)
    transformed_poses = np.array(transformed_poses_df)

    comb =[]
    
    for i in range(np.shape(transformed_points)[0]):
        placeholder = np.concatenate((transformed_points[i], transformed_poses[i]), axis=None)
        comb.append(placeholder)

    complete_poses_time = time.time()
    pose_transformation_time = complete_poses_time - subsample_and_poses_time_end
    print(f"pose transformation time:{pose_transformation_time}")
    
    plot= True
    if plot:
        fac = 10
        x1,y1,z1,x,y,z,w = [],[],[],[],[],[],[]
        for i in comb:
            x10,y10,z10,x0,y0,z0,w0 = i

            x1.append(x10)
            y1.append(y10)
            z1.append(z10)
            x.append(x0*fac+x10)
            y.append(y0*fac+y10)
            z.append(z0*fac+z10)


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x1, y1, z1, color='b', s=5)
        # ax.scatter(x, y, z, color='r', s=5, alpha=0.5) #used for more info in plot

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


    return np.array(comb)


#####
#########  MATH  ###########

def euclidian_distance(point):
    return np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)


########### Constants #############
    
camera_matrix= np.array([
    [-0.09208491, -0.9944386, -0.05110936, 175.9321],
    [0.9925474, -0.09578168, 0.07533592, -56.53379],
    [-0.07981229, -0.04379116, 0.9958475, -282.3045],
    [0,            0,          0,          1]])

posrot= np.array([726.694, 275.921, -601.061, 168.9561, -50.7837, -19.4969])#test0406-2

position= posrot_to_homogeneus(posrot)
# print(f"position = \n {np.array(position) }")
##


robot_matrix = position  #robot arm position


runde2:bool = False #NOTE set this to true to only run the short version


def main():
    start_time =time.time()

    # df= pd.read_csv('/home/zivid/Zivid/undistorted_results_sample.csv', sep = ',', header= None) #examples left over from earlier
    # df= pd.read_csv('Front2.csv', sep = ',', header= None)
    df= pd.read_csv('test-kevin.csv', sep = ',', header= None) #change this name based on the name of the file you want to read

    df_read_time_end= time.time()

    get_skeletonized_image_from_pointcloud(df, 
                                           [1944, 1200], 
                                           image_name_in= "/home/zivid/pytorch_env/LineDetection/images/results3.png",
                                           threshold=120, 
                                           save=True)
    HED_skeleton_time_end = time.time()
    get_skeletonized_image_from_pointcloud_canny(df, 
                                           [1944, 1200], 
                                           image_name_in= "/home/zivid/pytorch_env/LineDetection/images/results3.png",
                                           save=True)
    canny_time_end = time.time()
    print(f"HED time: {HED_skeleton_time_end- df_read_time_end}")
    print(f"canny time: {canny_time_end - HED_skeleton_time_end}")
    ####################################
    img_path = "/home/zivid/pytorch_env/skeleton_HED.png"
    skeleton= ski.io.imread(img_path, as_gray=True)

########################### template Matching if this is not the first iteration, here called "round 2"  #####################
    templatematching_start_time = time.time()
    if runde2: #change this later to set only this part to activate when re-running on a new image
        weld_paths= ["weld_path1.csv", "weld_path2.csv"]
        template_paths= ["template1.png","template2.png","template3.png"]
        # paths_templates = ["template_result"]
        apply_template_matching_automation(skel_image= skeleton, 
                                           template_path_paths=template_paths,
                                           weld_path_paths=weld_paths, 
                                           df=df, 
                                           matrix_rob=robot_matrix, 
                                           matrix_cam= camera_matrix)
        
        templatematching_end_time = time.time()

        print()
        print("Template matching:")
        df_read_time = df_read_time_end-start_time
        print(f"{df_read_time = }")

        skeleton_HED_time= HED_skeleton_time_end-df_read_time_end
        print(f"{skeleton_HED_time = }")

        canny_time = canny_time_end - HED_skeleton_time_end
        print(f"only: {canny_time = }")


        round2_time = templatematching_end_time - templatematching_start_time
        print(f"{round2_time = }")

        round2_total_time_HED = templatematching_end_time - start_time - canny_time
        print(f"{round2_total_time_HED = }")

        round2_total_time_Canny = templatematching_end_time - start_time - skeleton_HED_time
        print(f"{round2_total_time_Canny = }")

        exit()
    
    click_start_time= time.time()


    ########################USER INTERFACE START
    root = Tk()

    # setting up a tkinter canvas
    canvas = Canvas(root)

    # adding the image
    # img_path = "/home/zivid/pytorch_env/skeleton_HED.png"  
    img = Image.open(img_path)
    # width, height = img.width, img.height 
    img_tk = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, image=img_tk, anchor="nw")

    # Resize canvas to fit image
    canvas.config(width=img.width, height=img.height)

    # function to be called when mouse is clicked
    _coordinate_holder = []
    _bounding_box_center = []

    def left_click(event):
        # outputting x and y coords to console
        # cx, cy = event2canvas(event, canvas)
        # print ("(%d, %d) / (%d, %d)" % (event.x,event.y,cx,cy))
        _coordinate_holder.append([event.x, event.y])
        _bounding_box_center.append([event.x, event.y])

    
    def quit(event):
        print("Q has been pressed, exiting window")
        print("WARNING: This will cause the program to crash if insuficcient points are selected")
        root.destroy()
        
    def next(event):
        if len(_coordinate_holder) >=3 and len(_bounding_box_center)>=1:
            print()
            print(f"Sucsess! {len(_coordinate_holder)} points and {len(_bounding_box_center)} box centers extracted")
            print("going to next step")
            root.destroy()
        else:
            print()
            print(f"only {len(_coordinate_holder)} points {len(_bounding_box_center)} box centers chosen")
            print("3 points (left click) ")#, and 1 bounding box center (right click) are needed")
            print(f"Please choose the remaining ({max(3-len(_coordinate_holder),0)}) points and ({max(1-len(_bounding_box_center),0)}) box centers ")


    # mouseclick event
    canvas.bind("<ButtonPress-1>", left_click)
    # canvas.bind("<ButtonPress-3>", right_click)
    root.bind("<Return>", next)
    root.bind("q", quit)
### now to create the size of the bounding box and implement it

    canvas.pack()

    root.mainloop()

    #### EXITING THE TKinter LOOP ####
    ##################  USER INTERFACE STOP###############
    click_end_time= time.time()
    skeleton = ski.io.imread(img_path, as_gray=True)

    coordinate_temp = _coordinate_holder.copy()
    # print(_bounding_box_center)

    ##### creating three templates
    xy_n = []
    iter = 1
    _bounding_box_center = get_actual_cordinates(skeleton, _bounding_box_center) # set the template-matching points to be the same as the clicked ones
    for center in _bounding_box_center:
        xy= template_matching_extended(skeleton, center=center, iter=iter, size=400, show= False)
        # print(f"{xy = }")
        point = pixel_to_point(xy,df)
        # print(f"{point = }")
        transformed_point = transform_point(robot_matrix, camera_matrix, point)[0] #stupid that it returns a 2d list with only one element, but afraid to change due to other places it might be used
        # print(f"{transformed_point = }")

        xy_n.append(transformed_point)
        iter +=1
    # print(f"{np.array(xy_n) = }")
    save_point(np.array(xy_n), "template_points.csv") #saves the list of points as a csv


    actual_coordinates= get_actual_cordinates(skeleton, coordinate_temp)
    

#####################################################THE DOUBLE IMPLEMENTATION should be streamlined, but works for now
    placeholder_image = np.zeros_like(skeleton)
    placeholder_image1 = np.zeros_like(skeleton)
    placeholder_image2 = np.zeros_like(skeleton)
    # print(actual_coordinates[0:2])
    # print(actual_coordinates[1:3])
    coords1 = actual_coordinates[0:2]
    coords2 = actual_coordinates[1:3]
    lines_image = draw_lines(placeholder_image, actual_coordinates)
    lines_image1 = draw_lines(placeholder_image1, coords1)
    lines_image2= draw_lines(placeholder_image2, coords2)
    

    ski.io.imsave("weld_lines.png", placeholder_image, check_contrast=False)
    ski.io.imsave("weld_line1.png", placeholder_image1, check_contrast=False)
    ski.io.imsave("weld_line2.png", placeholder_image2, check_contrast=False)

    # lines_image_sorted= get_traversed_image(actual_coordinates, "weld_lines.png")
    lines_df = pd.DataFrame(lines_image1.flatten())
    lines_df2 = pd.DataFrame(lines_image2.flatten())
    df2 = df.copy()


        ####filter and subsample and pose##########
    df_filtered = filter_df_by_df(df,lines_df)
    df_filtered2 = filter_df_by_df(df2,lines_df2)


    sorted_lines= sort_linesegments(df_filtered, df_filtered2)
    df_sorted_lines1= pd.DataFrame(sorted_lines[0])
    df_sorted_lines2= pd.DataFrame(sorted_lines[1])
    df_sorted_lines_comb = pd.concat([df_sorted_lines1, df_sorted_lines2], axis=0, ignore_index=True)


    df_sorted_lines1.to_csv("weld_path1.csv", header = None, index = None) #Call this one 
    df_sorted_lines2.to_csv("weld_path2.csv", header = None, index = None)
    df_sorted_lines_comb.to_csv("weld_path.csv", header = None, index = None) #or this one in  do_subsample_extract_transform

    




        #### TRANSFORMATION ####

    create_combined_layer_visualization()# used to create the "...thin" and "...thick" images


    df_sorted_transformed_lines1= df_transformation(matrix_rob=robot_matrix, matrix_cam= camera_matrix, points= df_sorted_lines1) #pos01, camera_matrix, df_sorted_lines1)
    df_sorted_transformed_lines2= df_transformation(robot_matrix, camera_matrix, df_sorted_lines2)
    df_sorted_transformed_lines_comb = pd.concat([df_sorted_transformed_lines1, df_sorted_transformed_lines2], axis=0, ignore_index=True)

    df_sorted_transformed_lines1.to_csv("weld_path1_transformed.csv", header = None, index = None)
    df_sorted_transformed_lines2.to_csv("weld_path2_transformed.csv", header = None, index = None)
    df_sorted_transformed_lines_comb.to_csv("weld_path_transformed.csv", header = None, index = None)

        #### end of transformation ####
        #########The call to EE path subsampling and pose generation###########
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
    subsample_poses_time_start = time.time()
    # output = do_subsample_extract_transform(df_sorted_lines_comb, df, robot_matrix, camera_matrix, angle_offset=15, chosen_point_distance=10, pose_as_quaternion_xyzw= True)
    output = do_subsample_extract_transform(df_sorted_lines_comb, raw_to_xyz(df).round(3), robot_matrix, camera_matrix, angle_offset=30, chosen_point_distance=40, starting_point_distance=150,  pose_as_quaternion_xyzw= True)
    # output = do_subsample_extract_transform(df_sorted_lines_comb, raw_to_xyz(df).round(3), robot_matrix, camera_matrix, angle_offset=45, chosen_point_distance=40, starting_point_distance=150,  pose_as_quaternion_xyzw= True)

    # output = do_subsample_extract_transform(df_sorted_transformed_lines_comb.round(3), df_transformed, robot_matrix, camera_matrix, angle_offset=30, chosen_point_distance=40, pose_as_quaternion_xyzw= True)
    subsample_poses_time_end = time.time()

    out_df = pd.DataFrame(output)
    out_df.to_csv("OUTPUT_ORIGINAL.csv", header = None, index = None)# saving to two files as one will be altered by the template-matching
    out_df.to_csv("OUTPUT.csv", header = None, index = None)









    ####### time measurements
    end_time = time.time()
    print()
    df_read_time = df_read_time_end-start_time
    print(f"Initianl Pointcloud read time: {df_read_time}")

    HED_skel_time= HED_skeleton_time_end - df_read_time_end
    print(f"only: {HED_skel_time = }")

    canny_time = canny_time_end - HED_skeleton_time_end
    print(f"only: {canny_time = }")


    click_time = click_end_time - click_start_time
    print(f"GUI, The time taken on the GUI, mostly waiting for human input: {click_time = }")

    click_end_to_subsampling_time = subsample_poses_time_start - click_end_time
    print(f"time from completed GUI to subsampling starts: {click_end_to_subsampling_time}")

    path_poses_time = subsample_poses_time_end- subsample_poses_time_start
    print(f"complete time from subsampling starts to poses are extracted: {path_poses_time = }")


    total_second_part_time = end_time - click_end_time
    print(f"Total time used after GUI has ended (this is everything that DOSEN'T have to be done when template-matching): {total_second_part_time = }")


    total_time_with_click = end_time - start_time
    print(f"Total time for the whole process, including waiting for the GUI: {total_time_with_click}")

    total_time_without_click= df_read_time + HED_skel_time + total_second_part_time
    print(f"Total time from start to finish, with HED used, without GUI time included: {total_time_without_click}")

    total_time_without_click_canny= df_read_time + canny_time + total_second_part_time
    print(f"Total time from start to finish, with CANNY used, without GUI time included: {total_time_without_click_canny}")


if __name__ == "__main__":
    main()