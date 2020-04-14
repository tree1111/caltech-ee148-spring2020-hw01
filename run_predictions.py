import os
import numpy as np
import json
from PIL import Image

def normalization(I):
    I_norm = I.astype(float) * 2/ 255 - 1
    I_norm = I_norm / np.sqrt(np.sum(I_norm * I_norm))
    return I_norm

def smaller(I, k):
    indexm = np.arange(0, np.size(I, 0), k)
    indexn = np.arange(0, np.size(I, 1), k)
    I_smaller = I[indexm, :, :]
    I_smaller = I[:, indexn, :]
    I_smaller = normalization(I_smaller)
    return I_smaller

def fatter(I, k):
    m = np.size(I, 0)
    n = np.size(I, 1)
    I_bigger = np.zeros([m, k * n, 3])
    for j in range(3):
        I_bigger[:, :, j] = np.repeat(I[:, :, j], 2).reshape(m, k * n)

    I_bigger = normalization(I_bigger)
    return I_bigger

def match_conv(I, target, bounding_boxes, tao):
    '''
    This function computes convolution of the normalization target images and the
    images.
    '''
    box_height = np.size(target, 0)
    box_width = np.size(target, 1)
    height = np.size(I, 0)
    width = np.size(I, 1)
    M = height - box_height
    N = width - box_width
    conv = np.zeros([M, N])
    for m in range(M):
        for n in range(N):
            part = I[m:m+box_height, n:n+box_width, :]
            part = normalization(part)
            conv[m][n] = np.sum(target * part)
            if conv[m][n] > 0.93:
                box = [m, n, m+box_height, n+box_width]
                bounding_boxes.append(box)
                tao.append(conv[m][n])

    return bounding_boxes, tao

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

    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''
    tao = [] # This is the list of threshold of each box
    for k in range(4):
        if k == 0:
            target = np.array(Image.open('target/target_2.jpg'))
            target = normalization(target)
            bounding_boxes, tao = match_conv(I, target, bounding_boxes, tao)
        if k == 1:
            target = np.array(Image.open('target/target_2.jpg'))
            target = smaller(target, 2)
            bounding_boxes, tao = match_conv(I, target, bounding_boxes, tao)
        '''
        if k == 2:
            target = np.array(Image.open('target/target_2.jpg'))
            target = fatter(target, 2)
            bounding_boxes = match_conv(I, target, bounding_boxes)
        '''

        if k == 3:
            target = np.array(Image.open('target/target_1.jpg'))
            target = normalization(target)
            bounding_boxes, tao = match_conv(I, target, bounding_boxes, tao)
        '''
        if k == 4:
            target = np.array(Image.open('target/target_1.jpg'))
            target = fatter(target, 2)
            bounding_boxes = match_conv(I, target, bounding_boxes)
        if k == 5:
            target = np.array(Image.open('target/target_1.jpg'))
            target = fatter(target, 2)
            bounding_boxes = match_conv(I, target, bounding_boxes)
        '''
    num = len(bounding_boxes)
    flag = np.zeros(num)
    for index1, box1 in enumerate(bounding_boxes):
        if flag[index1] == 0:
            for index2, box2 in enumerate(bounding_boxes):
                if index1 != index2:
                    if flag[index2] == 0:
                        minX = max(box1[0], box2[0])
                        minY = max(box1[1], box2[1])
                        maxX = min(box1[2], box2[2])
                        maxY = min(box1[3], box2[3])
                        s1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                        s2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                        if minX<maxX and minY<maxY:
                            s = (maxX -minX) * (maxY - minY)
                            if s/s1 > 0.5 or s/s2 > 0.5:
                                if tao[index1] > tao[index2]:
                                    flag[index2] = 1
                                else:
                                    flag[index1] = 1

    bounding_boxes_final = []
    for n in range(num):
        if flag[n] == 0:
            bounding_boxes_final.append(bounding_boxes[n])

    bounding_boxes = bounding_boxes_final


    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes

# set the path to the downloaded data: 
data_path = 'RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = 'data/hw01_preds'
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}
for i in range(len(file_names)):
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    
    preds[file_names[i]] = detect_red_light(I)
    print(i)
    print('num is', len(preds[file_names[i]]))


# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
