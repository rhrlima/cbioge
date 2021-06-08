import cv2 
import glob
import numpy as np 
import os
import pickle
import re


def load_dataset(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def get_natural_key(string):
    matches = re.findall('(\\d+)', string)
    if len(matches) > 0:
        return int(matches[-1])
    else:
        return 0


def load_predictions(folder):
    files = glob.glob(os.path.join(folder, '*.png'))
    files.sort(key=lambda x: get_natural_key(x))
    images = [cv2.imread(f) for f in files]
    return np.array(images)


dataset = 'bsds500'
folder = 'unet_bsds'

# le o dataset  
dataset = load_dataset(f'datasets/{dataset}.pickle')
images = dataset['x_test']
labels = dataset['y_test']
labels = labels.astype('uint8')

# le as predicoes
preds = load_predictions(os.path.join('results', folder))

# amostra uma imagem
index = 10
i = images[index]
l = labels[index,:,:,0] * 255
p = preds[index]

cv2.imshow('Source', i)
cv2.imshow('Label', l)
cv2.imshow('Pred', p)


# # Let's load a simple image with 3 black squares 
# image = cv2.imread('analyze/10.png') 


# # Grayscale 
# gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray', gray)


edged = cv2.Canny(p, 100, 200)
cv2.imshow('Canny', edged)


# threshold
ret, thre = cv2.threshold(p, 100, 255, 0)
cv2.imshow('Threshold', thre)


# #contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# #cv2.imshow('Canny Edges After Contouring', contours)

# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))#np.ones((3,3), np.uint8)
# eroded = cv2.erode(img, kernel)
# cv2.imshow('Eroded', eroded)

# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('Closing', closing)

# # # Find Canny edges 
# # edged = cv2.Canny(gray, 100, 200)
# # #cv2.waitKey(0) 
  
# # # Finding Contours 
# # # Use a copy of the image e.g. edged.copy() 
# # # since findContours alters the image 
# # contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# # #contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

# # cv2.imshow('Canny Edges After Contouring', edged) 
# # #cv2.waitKey(0) 
  
# # print("Number of Contours found = " + str(len(contours)))
  
# # # Draw all contours 
# # # -1 signifies drawing all contours 
# # #temp = edged.copy()
# # cv2.drawContours(image, contours, -1, (255, 0, 255), 1)
  
# # cv2.imshow('Contours', image)
# # #cv2.waitKey(0) 

# # #temp = cv2.subtract(gray, image)
# # #temp = cv2.bitwise_and(gray, image)
# # # #Step 4: Erode the original image and refine the skeleton
# # # eroded = cv2.erode(img, element)
# # # skel = cv2.bitwise_or(skel,temp)

# # #cv2.imshow('Sub', temp) 
cv2.waitKey(0) 

cv2.destroyAllWindows()
