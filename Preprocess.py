#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract
from pytesseract import Output

# In[4]:

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def borders(here_img, thresh):
    size = here_img.shape
    check = int(115 * size[0] / 600)
    image = here_img[:]
    top, bottom = 0, size[0] - 1
    #plt.imshow(image)
    #plt.show()
    shape = size

    #find the background color for empty column
    bg = np.repeat(thresh, shape[1])
    count = 0
    for row in range(1, shape[0]):
        if  (np.equal(bg, image[row]).any()) == True:
            #print(count)
            count += 1
        else:
            count = 0
        if count >= check:
            top = row - check
            break
    
    
    shape = image.shape
    bg = np.repeat(thresh, shape[1])
    count = 0
    rows = np.arange(1, shape[0])
    #print(rows)
    for row in rows[::-1]:
        if  (np.equal(bg, image[row]).any()) == True:
            count += 1
        else:
            count = 0
        if count >= check:
            bottom = row + count
            break
    #print(count)
    
    
    #plt.imshow(here_img[top:bottom, :])
    #plt.imshow(here_img[top:bottom, :])
    #plt.show()
    
    d1 = (top - 2) >= 0 
    d2 = (bottom + 2) < size[0]
    d = d1 and d2
    if(d):
        b = 2
    else:
        b = 0
    
    return (top, bottom, b)


# In[5]:


def detect_text(main_image, gray_img, localized, bc):        
        cimg = cv2.resize(localized, (30, 30))
        bordersize = 1
        nimg = cv2.copyMakeBorder(cimg, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType=cv2.BORDER_CONSTANT, value=[255-bc, 0, 0])

        return main_image, nimg


# In[6]:


def segmentation(bordered, thresh):
    try:
        shape = bordered.shape
        check = int(50 * shape[0] / 320)
        image = bordered[:]
        image = image[check:].T
        shape = image.shape
        #plt.imshow(image)
        #plt.show()

        #find the background color for empty column
        bg = np.repeat(255 - thresh, shape[1])
        bg_keys = []
        for row in range(1, shape[0]):
            if  (np.equal(bg, image[row]).all()):
                bg_keys.append(row)            

        lenkeys = len(bg_keys)-1
        new_keys = [bg_keys[1], bg_keys[-1]]
        #print(lenkeys)
        for i in range(1, lenkeys):
            if (bg_keys[i+1] - bg_keys[i]) > check:
                new_keys.append(bg_keys[i])
                #print(i)

        new_keys = sorted(new_keys)
        #print(new_keys)
        segmented_templates = []
        first = 0
        for key in new_keys[1:]:
            segment = bordered.T[first:key]
            segmented_templates.append(segment.T)
            #show middle segments
            #plt.imshow(segment.T)
            #plt.show()
            first = key
        last_segment = bordered.T[new_keys[-1]:]
        segmented_templates.append(last_segment.T)
        
        #check if each segment shape is enough to do recognition
        

        return(segmented_templates)
    except:
        return [bordered]


# In[7]:


def localize(main_image, gray_img, localized, bc, show):
    #open the template as gray scale image
        template = localized
        #print(template.shape)
        width, height = template.shape[::-1] #get the width and height
        #match the template using cv2.matchTemplate
        match = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        position = np.where(match >= threshold) #get the location of template in the image
        for point in zip(*position[::-1]): #draw the rectangle around the matched template
            cv2.rectangle(main_image, point, (point[0] + width, point[1] + height), (255 - bc, 0, bc ), 2)

        return main_image
    


# In[8]:


def preprocess(bgr_img):#gray image   
    img = bgr_img[:]
    # d = pytesseract.image_to_data(img, output_type=Output.DICT)
    # n_boxes = len(d['level'])
    # for i in range(n_boxes):
    #     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret,th_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #converts black to white and inverse
    rows, cols = th_img.shape
    bg_test = np.array([th_img[i][i] for i in range(5)])
    if bg_test.all() == 0:
        text_color = 255
    else:
        text_color = 0
    #print('Process: Localization....\n')
    tb = borders(th_img, text_color)
    lr = borders(th_img.T, text_color)
    dummy = int(np.average((tb[2], lr[2]))) + 2
    template = th_img[tb[0]+dummy:tb[1]-dummy, lr[0]+dummy:lr[1]-dummy]
    #print("Process: Segmentation....\n")
    # th_img=image_resize(th_img, width=1200, height=1200)
    # template=image_resize(template,width=1200)
    # kernel = np.ones((5,5),np.uint8)
    # template = cv2.erode(template,kernel,iterations = 5)
    segments = segmentation(template, text_color)
    #print('Process: Detection.....\n')
    return segments, template, th_img, text_color

# In[ ]:




