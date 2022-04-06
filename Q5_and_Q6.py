import os
import numpy as np
import json
from PIL import Image

# code snippet from https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        # print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def get_radial_gradient(size):
    r = Image.radial_gradient(mode='L')
    radial_gradient = np.asarray(r.resize((size, size)))
    while np.sum(radial_gradient) > 0:
        radial_gradient = radial_gradient - np.ones((size, size))
    return radial_gradient

def get_feature_map(image):
    kernel = get_radial_gradient(10)
    blur_kernel = np.ones((5,5)) * (1 / 25)

    s = np.sum(image, axis=2)
    s[s==0] = 1
    color_map = (image[:, :, 0] / s) ** 2
    color_map = color_map / np.max(color_map) * 255
    feature_map = convolve2D(color_map, kernel, padding=10, strides=1)
    feature_map = feature_map / np.max(feature_map) * 250
    feature_map = convolve2D(feature_map, blur_kernel, padding=10, strides=1)
    return feature_map

def remove_similar_locations(locations):
    min_dist = 100
    centroids = [locations[0]]
    for loc in locations:
        is_new_centroid = True
        for c in centroids:
            if abs(loc[0] - c[0]) + abs(loc[1] - c[1]) < min_dist:
                is_new_centroid = False
        if is_new_centroid:
            centroids.append(loc)
    return centroids

def get_locations(feature_map):
    top_k = 20
    max_val = np.partition(feature_map.flatten(), -top_k)[-top_k]
    if max_val <= 100:
        return []
    feature_map[feature_map < max_val] = 0
    feature_map[feature_map >= max_val] = 255
    locs = (feature_map > 0).nonzero()
    return remove_similar_locations(list(zip(locs[0], locs[1])))

def get_bounding_boxes(locations):
    box_height = 50
    box_width = 50
    bounding_boxes = []
    for x, y in locations:
        tl_row = x - box_height / 2
        tl_col = y - box_width / 2
        br_row = tl_row + box_height / 2
        br_col = tl_col + box_width / 2
        bounding_boxes.append([tl_col, tl_row, br_col, br_row])
    return bounding_boxes


def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the
    image. Each element of <bounding_boxes> should itself be a list, containing
    four integers that specify a bounding box: the row and column index of the
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''


    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below.

    map = get_feature_map(I)
    locs = get_locations(map)
    bounding_boxes = get_bounding_boxes(locs)

    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4

    return bounding_boxes

# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# set a path for saving predictions:
preds_path = '../data/hw01_preds'
os.makedirs(preds_path,exist_ok=True) # create directory if needed

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

preds = {}
for i in [10, 46, 47, 332, 24, 33, 73, 90]:
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds[file_names[i]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
