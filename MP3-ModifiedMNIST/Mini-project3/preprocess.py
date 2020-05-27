"""
Find the bounding box of each digit and perform preprocessing
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

cv2.destroyAllWindows()
plt.close('all')

train_images_path, train_labels_path = ('input/train_images.pkl', 'input/train_labels.csv')

images = pd.read_pickle(train_images_path)[0:5,:,:]

labels = pd.read_csv(train_labels_path)

def max_bbox(images):
    new_images = np.zeros_like(images)
    cent_h, cent_v = images.shape[1]//2, images.shape[2]//2
    max_size = images.shape[1]*images.shape[2]
    
    
    for idx in range(0, images.shape[0]):
            
        im_gs = images[idx,:,:].astype(np.uint8)
        ret, thresh = cv2.threshold(im_gs, 245, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        max_rect = (0,0,0,0)
        max_contour = None
        area_i = 0
        for (i,c) in enumerate(contours):
            rect = cv2.boundingRect(c)
            x,y,w,h = rect
        #    area = cv2.contourArea(c)
            area = w + h #this actually performs better than w*h funny enough
            if area < 0.8*max_size and area > area_i:
                max_rect = rect
                area_i = area
                max_contour = c
                cont_idx = i
        
        mask = np.zeros_like(im_gs) # Create mask where white is what we want, black otherwise
        cv2.drawContours(mask, contours, cont_idx, 255, -1) # Draw filled contour in mask
        out = np.zeros_like(im_gs) # Extract out the object and place into output image
        out[mask == 255] = im_gs[mask == 255]
        new_images[idx, :,:] = out
            
        x,y,w,h = max_rect

    if idx%100 == 0:    
        print(idx)
    
    return new_images
    
### preprocess train data
train_images_path, train_labels_path = ('input/train_images.pkl', 'input/train_labels.csv')
train_images = pd.read_pickle(train_images_path)
labels = pd.read_csv(train_labels_path)

new_train_images = max_bbox(train_images)
np.save('input/bbox_train_contour.npy', new_train_images)
    
## preprocess test data
test_images_path = 'input/test_images.pkl'
test_images = pd.read_pickle(test_images_path)
new_test_images = max_bbox(test_images)
np.save('input/bbox_test_contour.npy', new_test_images)
