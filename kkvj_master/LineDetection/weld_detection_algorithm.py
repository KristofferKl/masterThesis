import numpy as np
import pandas as pd
from dataManipulation import raw_to_xyz


def mean(a,b):
    return [(a[0]+ b[0])/2, (a[1] +b[1])/2, (a[2] + b[2])/2]
def absolute(a):
    assert len(a) == 3, "not a list/vector/array of length 3"
    return np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)

def is_points_same(point1:list, point2:list):
    if point1[0] == point2[0] and point1[1] == point2[1] and point1[2] == point2[2]:
        return True
    return False

def is_point_in(point:list, in_this:list[list]):
    for npoint in in_this:
        if is_points_same(point, npoint):
            return True
    return False



def subsample_points(points:np.ndarray[list], threshold_weld_degrees= 30, threshold_noise_degrees= 75, handle_corner= False): #to handle the corner-rotation done in end_effector_pose_sorted_PC, one option is: make n large (or distance based) and calculate the direction vector from n distance on both sides of the point, or just n distance behind it. 
    """


# threshold_weld_degrees is usually set to 20 degrees change, might be too big, 0.1763 corresponds to 10 degrees offset, might be better, its the sine of the angle
# threshold_noise_degrees is usually set to 75 degrees, anything within this area is most likely some sort of noise, not an actual point
# points_IN = [[x,y,z],[x,y,z],[x,y,z], ...] # used to show how ther points are
# subsampl_IN = []
    """
    #this function looks n points forward
    # n = 10 # could make this a function based on npoints aswell, but want something kinda large to avoid problems with the coordinates alternating
    n = 100 
    sample_factor = 0.02119 # from thesting this is shown to give pretty good results
    if handle_corner:
        #n=150 works for + 2cm offset up
        n=180
        sample_factor = 0.04
    
    subsample = []
    vectors = []
    weld_index = []
    point_index_stack = []
    npoints = int(np.shape(points)[0] * sample_factor)
    # print(f"{np.shape(points)[0]//npoints = }")
    if npoints != 0:
        scalar = np.shape(points)[0]//npoints

        # print(f"{list(range(npoints)) = }")
        for i in range(npoints):
            point_index_stack.append(i*scalar)
        # point_index_stack = np.shape(points)[0]//npoints * list(range(npoints)) # finish this
        # print(f"{points[0:200:10] = }")
        # print(f"{len(point_index_stack)= }")
        if point_index_stack[0] == 0: #this should have a aditional check
            point_index_stack.pop(0)
        # print(f"{point_index_stack= }")

    subsample.append(points[0]) # adding first point to the subsample as the first ten points are not eglible to be chosen
    for i in range(np.shape(points)[0]):
        # print(f"{points[i]= }")
        if i + 2*n >= len(points)-1:
            break
        
        current = points[i]
        # print(f"{current= }")
        

        next_mean = mean(points[i+n-1], points[i+n]) #sums the value of each coordinate before dividing it by 2, returns the mean of the input points, goes to half the range of the currenntly viewed points
        current_vector = next_mean - current #could create a "current_mean" to make the results more accurate
        observed = points[i+2*n]
        observed_vector = observed - next_mean 
        # print(f"{current_vector= } ")
        current_vector_normalized = current_vector / absolute(current_vector)
        observed_vector_normalized = observed_vector / absolute(observed_vector)
        delta = absolute(np.cross(current_vector_normalized, observed_vector_normalized)) #only interested in the value, could potentially look at the direction but that is TODO
        # print(f"{absolute(current_vector)= }")
        # print(f"{delta= }")
        if len(point_index_stack):
            if i >= point_index_stack[0]:
                subsample.append(points[i+n])
                vectors.append(observed_vector_normalized)
                point_index_stack.pop(0) #removes the first ndex
                # print(f"{current_vector = }")
                # print(f"{observed_vector= }")
                # print()
                continue

        if not handle_corner:
            if np.arcsin(delta)*180/np.pi >= threshold_weld_degrees and np.arcsin(delta)*180/np.pi < threshold_noise_degrees:#making it into degrees
                subsample.append(points[i+n]) #observed
                vectors.append(observed_vector_normalized)
                weld_index.append(i) # can potentially be used to turn on/off the welding torch, not fully implemented yet
        if np.shape(vectors)[0]< np.shape(subsample)[0]:
            vectors.append(observed_vector_normalized) # shoul only be used in the first iteration, but makes sure there are a corresponding vector to each point
                
    #aditionally this can be included to make shure the last point of the line is included:
    last = points[-1]
    if not is_point_in(last, subsample): #evt: if last != subsample[-1]:
        subsample.append(last)
        vectors.append(vectors[-1]) #adding the last vector to have a corresponding vector to the last point
    #
    assert np.shape(subsample)[0]==np.shape(vectors)[0], f"Error: the output subsample {np.shape(subsample)} and vectors {np.shape(vectors)} does not have the same amount of datapoints"
    # print(f"{np.shape(subsample) = } \n {np.shape(vectors) = }")
    return subsample, vectors, weld_index



def main():
   df= pd.read_csv("weld_path1.csv", header=None)
   subsample, vectors, weld_index= subsample_points(np.array(raw_to_xyz(df)))
#    print(f"{np.shape(subsample)= }")
#    print(f"{np.array(vectors) = } ")
#    print(f"{len(subsample)= }, {len(vectors) = }")

   for i in range(len(vectors)):
       if i+1 >= len(vectors):
           break
       cross = np.cross(vectors[i], vectors[i+1])

    #    print(f"{cross/absolute(cross), absolute(cross)  = }")
       
   
   
   return

if __name__ == "__main__":
    main()