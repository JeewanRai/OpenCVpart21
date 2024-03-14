
# Watershed Algorithm 
Watershed is area where all water fall and snow melt are diverted down to streams, rivers and eventually to reservoirs, bays, and ocean.                    
Any grayscale image can be viewed as tropographical surface where high intensity denotes peaks and hills while low intensity denotes valleys. 
![alt text](image-116.png)

For human eyes the coins placed to each other might look separated but in computer vision use some kind of filtering it will see all the coins are connected with eath other. 
![alt text](image-117.png)

Physically they are separate coins but some computer vision algorithm we can see they are block of same image with white background.
![alt text](image-118.png)

Segment the image with 8 for the coins and 1 for the background.

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def display(image, cmap = 'gray'):
    fig = plt.figure(figsize = (12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap = 'gray')

coins = cv2.imread('coins.jpg')
display(coins)

# apply median blue, not concerned about drawing inside but
# concern with the circle of the coin only, convert to grayscale
# apply binary threshold
# find the contour. 
sep_blur = cv2.medianBlur(coins,21)
display(sep_blur)
```
Output:                             
![alt text](image-122.png)      
       ![alt text](image-123.png)

```Python
# converting to gray scale image
gray_sep_coins = cv2.cvtColor(sep_blur, cv2.COLOR_BGR2GRAY)
display(gray_sep_coins)
```
Output:                                     
![alt text](image-124.png)

```Python
# apply binary threshold for separating foreground and background
ret, sep_threshold = cv2.threshold(gray_sep_coins, 250, 255, cv2.THRESH_BINARY_INV)
display(sep_threshold)
```
Output:                                         
![alt text](image-126.png)

```Python
# Assuming sep_threshold is your binary image
contours, hierarchy = cv2.findContours(sep_threshold.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][2] == -1:  # external contour
        cv2.drawContours(coins, contours, i, (255, 0, 0), 10)

# Display the image with contours
display(coins)

```
The output of the image shows giant connected image at some points, need advance algorithm for separating.
Output:                             
![alt text](image-127.png)

```Python
image = cv2.imread('coins.jpg')

# apply large kernel size if the image size is big
image = cv2.medianBlur(image, 25)
display(image)
```
Output:                                         
![alt text](image-128.png)

```Python
# converting to gray scaleC
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
display(gray_image)
```
Output:                                                ![alt text](image-129.png)

```Python
# applying threshold
ret, thresh = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY_INV)
display(thresh)
```
We can find some noise in the image where inside of white coin there are few black spot; we will try Otsu's method which works well with watershed algorithm.
Output:                             
![alt text](image-130.png)

```Python
# applying threshold
ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# Noise removal
kernel = np.ones((3, 3), np.uint8)

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
display(opening)
```
Output:                                 
![alt text](image-131.png)

We can use distance transformation where the image at the center are brighter and image distance away from center appears lighter and lighter.
![alt text](image-132.png)
So we apply distance transformation on the thresh imae where it will appear brighter in the center and darker moving away from center since the coins are joined.
```Python
dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

display(dist_transform)
```
Output:                                     
![alt text](image-133.png)

```Python
# will apply another thresholding
ret, sur_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
display(sur_fg)
```
Output:                         
![alt text](image-134.png)

```Python
# display and find unknown regions
sur_fg = np.uint8(sur_fg)

unknown = cv2.subtract(sur_bg, sur_fg)
display(unknown)
```
Output:                                 
![alt text](image-135.png)

Creating level marker
```Python
# creating level marker
ret, marker = cv2.connectedComponents(sur_fg)
marker
```
Output:                             
```Python
array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]], dtype=int32)
```
```Python
# creating level marker
ret, marker = cv2.connectedComponents(sur_fg)
marker = marker + 1 # done to distinguish marker from background

marker[unknown == 255] = 0
display(marker)
```
Output:                                 
![alt text](image-136.png)

```Python
# apply watershed algorithm
marker = cv2.watershed(image, marker)
display(marker)
```
From the given output we can find different coin size with differe colored 
Output                                          
![alt text](image-137.png)

```Python
# Assuming sep_threshold is your binary image
contours, hierarchy = cv2.findContours(marker.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][2] == -1:  # external contour
        cv2.drawContours(coins, contours, i, (255, 0, 0), 10)

# Display the image with contours
display(coins)
```
We some how able to separate the each individual coin from each other

Output:                                             
![alt text](image-138.png)

## Customer Seeds with Watershed Algorithm
Automatically the different area are segmented whith mouse click.
![alt text](image-139.png)

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

road = cv2.imread('road.jpg')
road_copy = np.copy(road)

plt.imshow(road_copy)
```
Ouput:                          
![alt text](image-140.png)

```Python
# create empty space for result to be drawn
road_copy.shape
```
Output:                             
```Python
(650, 433, 3)
```
```Python
# create empty space for result to be drawn
# not required color channel so we use :2 for extracting
# height and weidth
road_copy.shape[:2]
```
Output:                             
```Python
(650, 433)
```

```Python
marker_image = np.zeros(road.shape[:2], dtype=np.int32)

# segment
segments = np.zeros(road.shape, dtype=np.uint8)

marker_image.shape
segments.shape
```
Output:                                     
```Python
(650, 433)
(650, 433, 3)
```

```Python
from matplotlib import cm

cm.tab10(0)
```
Result is RGB color with alpha parameter ranges from 0 and 1
Output:                                     
```Python
(0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0)
```
```Python
# converting to to 0 to 255
tuple(np.array(cm.tab10(0)[:3])*255)
```
Output:                                         
```Python
(31.0, 119.0, 180.0)
```
```Python
def create_rgb(i):
    return tuple(np.array(cm.tab10(i)[:3])*255)

colors = []
for i in range(10):
  colors.append(create_rgb(i))

colors = []
for i in range(10):
  colors.append(create_rgb(i))
```
Output:                                                     
```Python
[(31.0, 119.0, 180.0),
 (255.0, 127.0, 14.0),
 (44.0, 160.0, 44.0),
 (214.0, 39.0, 40.0),
 (148.0, 103.0, 189.0),
 (140.0, 86.0, 75.0),
 (227.0, 119.0, 194.0),
 (127.0, 127.0, 127.0),
 (188.0, 189.0, 34.0),
 (23.0, 190.0, 207.0)]
```
```Python
# set up callback function
# global variables
# call back function
# while true
n_markers = 10 # 0 to 9
current_marker = 1 # index position
mark_updates = False # checks markers has been updated

def mouse_callback(event, x, y, flags, param):
    global mark_updates

    if event == cv2.EVENT_LBUTTONDOWN:
        # draw 2 circles one for watershed and one for human visulization
        # marker passed to watershed algorithm
        cv2.circle(marker_image, (x, y), 10, (current_marker), -1)


        # user sees
        cv2.circle(road_copy, (x, y), 10, colors[current_marker], -1)
        mark_updates = True

cv2.namedWindow('Road Image')
cv2.setMouseCallback('Road Image', mouse_callback)

while True:
    cv2.imshow('Watershed Segments', segments)
    cv2.imshow('Road Image',road_copy)  

    #close all windows, upade color choice, update the marking
    k = cv2.waitKey(1)
    if k == 27:
        break
    # clearning all the colors press c key
    # resets images 
    elif k == ord('c'):
        road_copy = road.copy()
        marker_image = np.zeros(road.shape[:2], dtype=np.int32)
        segments = np.zeros(road.shape, dtype=np.uint8)
    # update color choice
    elif k > 0 and chr(k).isdigit():
        current_marker = int(chr(k))

    if mark_updates:
        marker_image_copy = marker_image.copy()
        cv2.watershed(road, marker_image_copy)

        segments = np.zeros(road.shape, dtype=np.uint8)
        for color_index in range(n_markers):
            # coloring segments using numpy call
            segments[marker_image_copy == (color_index)] = colors[color_index]

cv2.destroyAllWindows()
````
Output:                                         
![alt text](image-141.png)