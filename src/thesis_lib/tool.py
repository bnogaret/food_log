import os

from skimage.transform import resize

def read_bb_info_txt(path, d):
    """
    Read the bb_info.txt to get the rectangle coordinates.
    Add the coordinates of the rectangles into the dictionnary d.
    The name of the file is used as the key, the value is a list of 4 integers:
    - first and second value: coordinate of one of the corner
    - third and forth value: coordinate of the opposte corner
    """
    label = os.path.split(os.path.dirname(path))[1]
    with open(path, 'r') as f:
        f.readline() # skip the first line
        for line in f:
            # print(line)
            data = line.split()
            if data[0] in d:
                print("Multiple image for {}".format(data[0]))
                d[data[0]].append([int(i) for i in data[1:]])
            else:
                d[data[0]] = [[int(i) for i in data[1:]]]
                
def get_coordinate_resized_rectangles(img_shape, resized_shape, rectangles):
    x_scale = resized_shape[0]/img_shape[1]
    y_scale = resized_shape[1]/img_shape[0]
    resized_rectangles = []
    for rectangle in rectangles:
        resized_rectangles.append([
                int(rectangle[0] * x_scale),
                int(rectangle[1] * y_scale),
                int(rectangle[2] * x_scale), 
                int(rectangle[3] * y_scale)
                ])
    return resized_rectangles