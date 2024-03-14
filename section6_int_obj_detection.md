# Section 6 Object detection 
![alt text](image-52.png)
![alt text](image-53.png)
![alt text](image-54.png)
![alt text](image-55.png)
![alt text](image-56.png)
![alt text](image-57.png)

## Template Matching
Simplest form of Object Detection.
Template matching is image processing technique to match small image(template) which is part of large image with the large image.

This technique is widely used for object detection projects, like product quality, vehicle tracking, robotics etc

#### Analogy
Its like specific artical or image from a newspaper page, say finding suduko from larger news paper page which involve scanning through entire newspaper page and compare each section with smaller image or specific artical/template. The goal is to find regions where the contenet of the newpaper page closely matches the desired template. 
During template matching, a correlation map is created. It's like keeping track of how well the content of the newspaper page matches your template at different locations. High values in the correlation map indicate strong matches. Once the template matching process is complete, you can identify the location where the correlation is highest. This location corresponds to the position on the newspaper page where your template (specific article or image) is located.

The goal is to find regions in the input image where the template closely matches. The function slides the template over the input image and computes a correlation map (result) at each position, indicating how well the template matches the corresponding region in the input image. The resulting correlation map (result) will have higher values at positions where the template is a close match, and lower values where there is less similarity. This map is then analyzed to find the location of the best match, and a rectangle is often drawn around that region to highlight the detected template in the original image.


```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('card-2.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
```

Output:                         
![alt text](image-58.png)

Its part of the larger image **card-2**. The lareger image size is more than image size of head image. It will scann through each pixel of the image and find the match. 
```Python
head = cv2.imread('card-3.png')

head = cv2.cvtColor(head, cv2.COLOR_BGR2RGB)
plt.imshow(head)
```
Output:                      ![alt text](image-59.png)

Size difference between image and shape image
```Pytho
image.shape

head.shape
```
Output:                 
```Python
(435, 580, 3)

(80, 89, 3)
```

With method considered, heatmap is generated based on higher degree of correlation we find after scanning between main image and template image or shows where maximum values are matched.                  
The **eval** takes string say "5+8" and convert it to python form of 5 + 8 to perform python task such as arithmatic operation etc.resulting 13 as output.
```Python
my_method = eval('cv2.TM_CCOEFF')
res = cv2.matchTemplate(image, head, my_method)

plt.imshow(res)
```
Output:                             
![alt text](image-60.png)

Example or explination
```Python
# Define a mathematical expression as a string
expression = "5 + 3"

# Use eval to evaluate the string as a Python expression
result = eval(expression)

# Print the result
print("The result of the expression", expression, "is:", result)
```
Output:                             
```Python
The result of the expression 5 + 3 is: 8
```
We will find max and min values of the heatmap, max and min value location and then use that to draw red rectangle around the match of the template.

```Python 
# 6 methods for comparison in a list
# line initializes a list called methods containing strings representing different template matching techniques in OpenCV, including correlation methods
methods =['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR-NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for method in methods:
     # creating copy of original image
    image_copy = image.copy()
    
    technique = eval(method)

    #template matching
    # emplate matching is performed using the specified technique, comparing the input image (image) with the template image (head). The result is a correlation map stored in the variable result
    result = cv2.matchTemplate(image, head, technique)

    # Tuple unpacking use for find highest similarity and least similarity (min_val) and finding the location
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# checking if technique is equal to either cv2.TM_SQDIFF or cv2.TM_SQDIFF_NORMED. if its true then set top_left where there is minimum matches and vice versa
    if technique in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
     
    height, width, channels = head.shape
    
    bottom_right = (top_left[0] + width, top_left[1]+height)
    cv2.rectangle(image_copy, top_left, bottom_right, color=(255, 0, 0), thickness= 10 )

    plt.subplot(121)
    plt.imshow(result)

    plt.title('Template Matching')
    plt.subplot(122)
    plt.imshow(image_copy)
    plt.title('Detection of Template')
    plt.suptitle(method)

    plt.show()

    print('\n')
    print('\n')
```
Output:                             
![](image-61.png)
![alt text](image-62.png)

Did not find the correct match
![alt text](image-63.png)

![alt text](image-64.png)
![alt text](image-65.png)
![alt text](image-66.png)

Method 2 without uning eval function
```Python
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for method in methods:
    # creating a copy of the original image
    image_copy = image.copy()

    # template matching using the correct technique
    result = cv2.matchTemplate(image, head, method)

    # Tuple unpacking
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    height, width, channels = head.shape

    bottom_right = (top_left[0] + width, top_left[1] + height)
    cv2.rectangle(image_copy, top_left, bottom_right, color=(255, 0, 0), thickness=10)

    plt.subplot(121)
    plt.imshow(result)

    plt.title('Template Matching')
    plt.subplot(122)
    plt.imshow(image_copy)
    plt.title('Detection of Template')
    plt.suptitle(method)

    plt.show()

    print('\n')
    print('\n')
```

For Methods like cv2.TM_SQDIFF and cv2.TM_SQDIFF_NORMED:

These methods focus on finding the smallest differences between the template and the image.
In this case, the smaller the value, the better the match. The minimum value in the correlation map indicates the location where the template is the most similar to the image.
Therefore, the top-left corner of the rectangle is set to the location of the minimum value in the correlation map.
For Other Methods like cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED:

These methods focus on finding the largest correlations between the template and the image.
In this case, the larger the value, the better the match. The maximum value in the correlation map indicates the location where the template is the most similar to the image.
Therefore, the top-left corner of the rectangle is set to the location of the maximum value in the correlation map.

## Corner Detection
Corners is junction of two edges.
Popular corner detection algorithms:
Harris Corner Detection
Shi-Tomasi Corner Detection
![alt text](image-67.png)
![alt text](image-68.png)
![alt text](image-69.png)

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

flat_chess = cv2.imread('chess.jpg')
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)
plt.imshow(flat_chess)
```
Output:             
![alt text]
(image-70.png)

```Python
gray_falt_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_falt_chess, cmap ='gray')
```
![alt text](image-71.png)

```Python
real_chess = cv2.imread('real_chess.jpg')
real_chess =cv2.cvtColor(real_chess, cv2.COLOR_RGB2BGR)
plt.imshow(real_chess)
```
Output:                         
![alt text](image-72.png)

#### Gray scale form of the image
```Python
gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_RGB2GRAY)
plt.imshow(gray_falt_chess, cmap = 'gray')
```
Output:                 
![alt text](image-74.png)

#### Applying Harris corner detection
For Harris corner detection the value should be in float, right now we have only integer value.
```Python
float_gray = np.float32(gray_falt_chess)
float_gray
```
Output:                 
```Python
array([[255., 255., 255., ..., 255., 255., 255.],
       [255., 255., 255., ..., 255., 255., 255.],
       [255., 255., 255., ..., 255., 255., 255.],
       ...,
       [255., 255., 255., ..., 255., 255., 255.],
       [255., 255., 255., ..., 255., 255., 255.],
       [255., 255., 255., ..., 255., 255., 255.]], dtype=float32)
```
```Python
dest_var_name = cv2.cornerHarris(src=float_gray, blockSize=2, ksize=3, k=0.04)
dest_var_name = cv2.dilate(dest_var_name, None)
flat_chess[dest_var_name>0.01*dest_var_name.max()] =[0, 0, 255]
plt.imshow(flat_chess)
```
Output:                         
![alt text](image-75.png)

```Python
real_chessboard = np.float32(gray_real_chess)
dst = cv2.cornerHarris(src=real_chessboard, blockSize=2, ksize=3, k=0.04)

dst = cv2.dilate(dst, None)

real_chess[dst > 0.01 * dst.max()] = [255, 0, 0]
plt.imshow(real_chess)
```
Output:                                 
![alt text](image-76.png)

### Shi-Tomasi Corner Detection
Unlike Harris Corner Detection it will not detect all the corners but in this method we have to flatten the image to detect the corners.
```Python
flat_chess = cv2.imread('chess.jpg')
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)

real_chess = cv2.imread('real_chess.jpg')
real_chess =cv2.cvtColor(real_chess, cv2.COLOR_RGB2BGR)

gray_falt_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)
gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_RGB2GRAY)

corners = cv2.goodFeaturesToTrack(gray_falt_chess, 64, 0.01, 10)

corners =np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(flat_chess, (x,y), 3, (0, 255, 0), -1)

plt.imshow(flat_chess)
```
Output:                         
![alt text](image-77.png)

```Python
corners_1 = cv2.goodFeaturesToTrack(gray_real_chess, 80, 0.01, 10)

corners_1 = np.int0(corners_1)

for i in corners_1:
    x, y = i.ravel()
    cv2.circle(real_chess, (x,y), 3, (255, 0, 255), -1)
plt.imshow(real_chess)
```
Output:                             
![alt text](image-78.png)

## Edge Detection
Canny Edge detector, multi-stage algorithm
First step is to Apply Gaussian Filter to smooth the image and remove noise.            
Find the intensity gradients of the image.
Apply non-maximum supression to get rid of spurious response to edge detection
Apply double threshold to determine potential edges
Track edges by hysteresis, detection of edges by supressiong all the other edges that are weak and not connected to strong edges or process of filtering the weaker edges and only keeping the strong edges. 

```Python
# reading an image 
img = cv2.imread('card-2.jpg')
plt.imshow(img)
```
Output:                                 

![alt text](image-79.png)
```Python

# set the threshold value initially at 127
edges = cv2.Canny(image=img, threshold1=127, threshold2=127)
plt.imshow(edges)
```
From the given image we can find some edges but also has capture some of the external edges which is probably noise.
Output:                         
![alt text](image-80.png)

To solve the issue we can take two approaches where one is to blurring the image to eliminate certain details and other is to play with thresholds

```Python
# set the threshold value initially at 127
edges = cv2.Canny(image=img, threshold1=0, threshold2=255)
plt.imshow(edges)
```
With this setting we have picked up more details of the image.
Output:                             
![alt text](image-82.png)

We have formula to to find the required value of the pixel
```Python
# setting up lower and upper bound threshold
# Lower threshold to either 0 or 70% of the median value which ever is greater
lower = int(max(0, 0.7 * med_val))
# upper threshold to either 130% of the median or the max 255, which ever is smaller
upper =  int(min(255, 1.3 * med_val))

edges = cv2.Canny(image=img, threshold1=lower, threshold2 = upper)
plt.imshow(edges)
```
capturing more details of the edges.
Output:                             
![alt text](image-83.png)

So trying to adjust the value of the upper bound will somewhat improve edge detection but it will not resolve the issue so better to opt for bluring the image option
```Python
edges = cv2.Canny(image=img, threshold1=lower, threshold2 = upper + 100)
plt.imshow(edges)
```                  
Output:                                 
![alt text](image-84.png)

```Python
blur_img = cv2.blur(img, ksize = (5, 5))

edges = cv2.Canny(image=blur_img, threshold1=lower, threshold2 = upper)
plt.imshow(edges)
```
Now we can see stronger edges of the dog face
![alt text](image-85.png)

### Grid Detection
Often camerascreates distortion in image such as radial distortion  andtangential distortion

```Python
chess = cv2.imread('chess1.jpg')
plt.imshow(chess)

found, corners = cv2.findChessboardCorners(chess, (7, 7))

found
```
Output:   
found if its true corner is detected.
![alt text](image-86.png)

courners            
![alt text](image-87.png)

```Python
cv2.drawChessboardCorners(chess, (7,7), corners, found)

plt.imshow(chess)
```

Output:                 
![alt text](image-88.png)

```Python
dotgrid = cv2.imread('squaredotgrid.jpg')
plt.imshow(dotgrid)

found, corner = cv2.findCirclesGrid(dotgrid, (10, 10), cv2.CALIB_CB_SYMMETRIC_GRID)
found

corner
```
Output:                         
![alt text](image-89.png)

```Python
cv2.drawChessboardCorners(dotgrid, (10,10), corner, found)
plt.imshow(dotgrid)
```
this method is used for camera caliberation (grid detection method)
Ouput:                                  
![alt text](image-90.png)


### Contours Detection
A curve joining all the continuous points having same color or intensity. Useful for shape analysis and object detection and recognization. 

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('download.png', 0)

img.shape

plt.imshow(img, cmap='gray')
```
Output:                                 
```Python
(101, 501)
```
![alt text](image-92.png)

Random changes is called gradient
Contour is boundry around something with well defined edges. Has well defined edges so machine is able to compute difference in gradient and form recognisable shape thorugh continuing changes and draw boundry.

Contour is finding boundry of an image, traced along the edges, the change is pixel intensity between neighbouring pixel marks the boundy.

![alt text](image-93.png)

```Python
img = cv2.imread('thumbs_up_down.jpg', 0)
plt.imshow(img, cmap='gray')
```
Output                                  
![alt text](image-94.png)

Converting color image from RGB to RBG and to grayscale
```Python
# converting into RGB and then grayscale
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# conveting to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
```

need to create a binary image, which means each pixel of the image is either black or white. This is a necessity in OpenCV, finding contours is like finding a white object from a black background, objects to be found should be white and the background should be black.
```Python
# create a binary thresholded image
_, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
# show it
plt.imshow(binary, cmap="gray")
plt.show()
plt.show()
```
Ouput:                          
![alt text](image-95.png)

```Python
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img1 = cv2.drawContours(image, contours, -1, (255, 0, 0), 5)

# show the image with the drawn contours
plt.imshow(image)
plt.show()
```
Output:                                     
![alt text](image-96.png)

Examples of contour detection include foreground extraction, image segmentation, detection and recognization. In  contour detection we always use binary image 0 or 255. It cannot be use with grayscale image since in grayscale image we do not find clear edges unlike in binary image.

![alt text](image-97.png)
![alt text](image-98.png)

```Python
#read image, resize and convert to grayscale image
image = cv2.imread('images.jpeg')

image = cv2.resize(image, None, fx=0.9, fy=0.9)
# converting image for pyplot matching
plt_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(plt_img, cv2.COLOR_RGB2GRAY)

plt.imshow(gray, cmap='gray')
```

Output:                     
![alt text](image-99.png)

```Python
# converting grayscale image to binary image
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(binary)

# now detect contoures
contours, hierarchy = cv2.findContours(binary, mode =cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)

print("length of contours {}".format(len(contours)))
print(contours)

# draw contour ont he original image
image_copy = image.copy()
image_copy = cv2.drawContours(image_copy, contours, -1, (255, 0, 0 ), thickness=3, lineType=cv2.LINE_AA)


# visualizing the results
cv2.imshow('Grayscale Image', gray)
cv2.imshow('Draw Contours', image_copy)
cv2.imshow('Binary Image', binary)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
Output:                                         ![alt text](image-100.png)

### Contour Retreval Mode
Suppose an image has multiple contour in it so how will we report which one is first and which one is second in row to report. 
Retrieve external will extract only the contour of external item if there is contour inside of outercontoure it will ignore
Retrieve List is used when we want all the contour but ignore hierarchy(indicates this data belongs to that image and so on)

Retrive Tree its like ouput all the contours by establishing hierarchiercy relationship
![alt text](image-101.png)

In this code where ith hierarchical level indicates the index of next contour

![alt text](image-102.png)

![alt text](image-103.png)

## Feature Matching Part One
Unlike template matching which require same image say of dog's face but in real life its always not the same dog so have to extract some attributes, features of the dog using ideas from conrner,edge and contour detection. Using  distance calculation, finds all the matches in a secondaty image wihtout no loger requiremet to have exact copy of the tragated image.                         
Exploring following 3 methods:                          
    1.  Brute-Force Matching with ORB Descriptors           
    2.  Brute-Force Matching with SIFT Descriptors and Ratio Test                                       
    3.  FLANN based Matcher      

#### Brute-Force Matching with ORB (Orinted FAST and Rotated BRIEF)
Comparing features(numerical values) between first image and second image by finding difference between the images to discover similarity or dissimilarity.      

For BF matcher, first we have to create the BFMatcher object using cv.BFMatcher(). It takes two optional params. First one is normType. It specifies the distance measurement to be used. By default, it is cv.NORM_L2. It is good for SIFT, SURF etc (cv.NORM_L1 is also there). For binary string based descriptors like ORB, BRIEF, BRISK etc, cv.NORM_HAMMING should be used, which used Hamming distance as measurement. If ORB is using WTA_K == 3 or 4, cv.NORM_HAMMING2 should be used.

Second param is boolean variable, crossCheck which is false by default. If it is true, Matcher returns only those matches with value (i,j) such that i-th descriptor in set A has j-th descriptor in set B as the best match and vice-versa. That is, the two features in both sets should match each other. It provides consistent result, and is a good alternative to ratio test proposed by D.Lowe in SIFT paper.

Once it is created, two important methods are BFMatcher.match() and BFMatcher.knnMatch(). First one returns the best match. Second method returns k best matches where k is specified by the user. It may be useful when we need to do additional work on that.

Like we used cv.drawKeypoints() to draw keypoints, cv.drawMatches() helps us to draw the matches. It stacks two images horizontally and draw lines from first image to second image showing best matches. There is also cv.drawMatchesKnn which draws all the k best matches. If k=2, it will draw two match-lines for each keypoint. So we have to pass a mask if we want to selectively draw it.

Here, we will see a simple example on how to match features between two images. In this case, I have a queryImage and a trainImage. We will try to find the queryImage in trainImage using feature matching.
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
```
```Python
reeses = cv2.imread('snapshotimagehandler_1932655075.jpeg', 0)
display(reeses)
```
Output:                                             
![alt text](image-104.png)

```Python
ressesfuff = cv2.imread('ressesspufff.jpeg', 0)
display(ressesfuff)
```                                             
Output:                                         
![alt text](image-105.png)                          

```Python
# creating detector object/ initating orb detector 
orb = cv2.ORB_create()

# not masking so we use None in the second parameter
kp1, des1 = orb.detectAndCompute(reeses, None)

# creating detector object
orb = cv2.ORB_create()

# not masking so we use None in the second parameter
kp2, des2 = orb.detectAndCompute(ressesfuff, None)  

# creating matching object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

single_match = matches[0]
single_match.distance

matches = sorted(matches, key=lambda x:x.distance)

len(matches)
#only matching 25 key features of the given feature out of 114 matches
reeses_matches = cv2.drawMatches(reeses, kp1, ressesfuff, kp2, matches[:25], None, flags =2)

display(reeses_matches)
```

Output:                                             
![alt text](image-106.png)                          

## Featire Matching Part two
SIFT(Scale Invariant Feature Transform) Descriptors stands for scale and variant feature transform, does well when image size of different scale. In the earlier image the reese's puffs cereal box size of tagated image and real image was different which will cause some confusion.  

In this method suppose if we use template matching its quite tedious since we need to create multiple templates. Additionally, the image is obstracted with other image, also the training image is rotated in certain angle, so using template matching its difficult. Therefore, to detect features of querrying image from traning image we use SIFT Detectors where it will find the unique features, match them.

Corner, edge and contour detection is not always interest point for some task we required more information, description on the onject under consideration.  Detecting blobs which is some local appearance which gives description of the image. Object can appear in many different size, depth and magnification depending on camera so have to deal with scaling

The given image which has two setting where one is more clear than other, orintation are different and light intensity are different, as well the size are different. So its would be earier if we can reointate to match the image, or increase bightness of image and rescale to match the other image. Also, we should consider unwanted features like the background of the image. 
![alt text](image-112.png)
So the image content such as brightness, color, should have well degined feature or unique feature or signature feature, the position of the image is well defined, should not get hampared with image size and rotation and should not be affected by lights intensity changes. 
![alt text](image-113.png)
For the given image  intert point we have is the edges are show but if we go along the line we can again seeanother are of similar edges, which indictes the edges are not unique; corners are for simpler object.
![alt text](image-114.png)
From the given image we can see that blobs has different range of light intensity which are unique.

To use blobs like features we need to have certain attribues: location of blobs which does not depend on the image features. Determine the size is rough scale area bounding the area of interest, not associated with edges or bounderis of the object, its rough scale. Orientation and formulate the description or signature that is independed of the size and orientaiton
![alt text](image-115.png)


```Python
# create object 
sift = cv2.SIFT_create()

kp1, ds1 = sift.detectAndCompute(reeses, None)
kp2, ds2 = sift.detectAndCompute(ressesfuff, None)

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2)

matches 
```
Output:                                 
![alt text](image-107.png)

```Python
# apply ratio test for testing if two matches are relatively cloose to eath other
good =[] # less distance means better matches and vice versa
# if match1 distance is less than 75% of match 2 distance
# then descriptor was a good match, lets keep it
for match1, match2 in matches:
    if match1.distance < 0.75 * match2.distance:
        good.append([match1])

len(good)

len(matches)

sift_matches = cv2.drawMatchesKnn(reeses, kp1, ressesfuff, kp2,good, None, flags=2)
display(sift_matches)
```
```Python
2                                                   
500
```
![alt text](image-108.png)          

### FLANN Based Matches
```Python 
sift = cv2.SIFT_create()

kp1, ds1 = sift.detectAndCompute(reeses, None)
kp2, ds2 = sift.detectAndCompute(ressesfuff, None)

#Defining Fast Library Approximation Near Neighbour 
# faster then bruforce method but finds only general feature matches

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, tree = 5)
search_params = dict(checks = 1)


# Convert descriptors to float32 before matching
if des1.dtype != 'float32':
    des1 = des1.astype('float32')
if des2.dtype != 'float32':
    des2 = des2.astype('float32')

# Now you can use FLANN matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# ratio test
good = []

for match1, match2 in matches:
    if match1.distance < 0.7 * match2.distance:
        good.append([match1])

flann_matches = cv2.drawMatchesKnn(reeses, kp1, ressesfuff, kp2, good, None, flags=0)
display(flann_matches)

```
Output:                     
![alt text](image-109.png)

Addition additional features
```Python
matchesMask = [[0, 0] for i in range(len(matches))]

for i, (match1, match2) in enumerate(matches):
    if match1.distance < 0.7 * match2.distance:
        matchesMask[i] = [1, 0]  # Label lines

draw_params = dict(
    matchColor=(0, 255, 0),
    singlePointColor=(255, 0, 0),
    matchesMask=matchesMask,
    flags=0  # 'flag' parameter was renamed to 'flags'
)

# Assuming 'display' is a valid function for displaying images
flann_matches = cv2.drawMatchesKnn(reeses, kp1, ressesfuff, kp2, matches, None, **draw_params)
display(flann_matches)

```
Output:                                     
![alt text](image-111.png)

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
Output:                             
![alt text](image-127.png)


