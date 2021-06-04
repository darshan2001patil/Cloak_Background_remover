# Import Libraries
import cv2
import numpy as np
import time

##*********************************************************##
#
# 1. Write the Video in an output file
#
##*********************************************************##
cap = cv2.VideoCapture( 0 )
# Define the codec and create VideoWriter object
width = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) + 0.5 )
height = int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) + 0.5 )
size = (width, height)
fourcc = cv2.VideoWriter_fourcc( *'XVID' )
out = cv2.VideoWriter( 'output.avi', fourcc, 20.0, size )

##*********************************************************##
#
# 2. Capture, and store the background frame:
#
##*********************************************************##    
# 1. read from the webcam
# cap = cv2.VideoCapture(0)

# 2.give the camera some time to warm up
time.sleep( 3 )
count = 0
background = 0

# 3. Capture the background in range of 60
for i in range( 60 ):
    ret, background = cap.read()
    if ret == False:
        continue
background = np.flip( background, axis=1 )  # flipping the frame

##*********************************************************##
#
# 3. Detect the red colored cloak using color detection
#   and segmentation algorithm.
#
##*********************************************************##

## Read every frame from the webcam
while (cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        print( "Can't receive image (stream end?). Exiting ..." )
        break
    count += 1
    img = np.flip( img, axis=1 )

    ## Convert the color space from BGR to HSV
    hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV )

    ##*********************************************************##
    #
    # 4. Generate masks to detect red color
    #
    ##*********************************************************##

    # Defining lower range for red color detection.
    lower_red = np.array( [0, 120, 70] )
    upper_red = np.array( [10, 255, 255] )
    # lower_red = np.array([100, 40, 40])
    # upper_red = np.array([100, 255, 255])
    mask1 = cv2.inRange( hsv, lower_red, upper_red )

    # Defining upper range for red color detection
    lower_red = np.array( [170, 120, 70] )
    upper_red = np.array( [180, 255, 255] )
    # lower_red = np.array([170, 120, 70])
    # upper_red = np.array([10, 255, 255])
    mask2 = cv2.inRange( hsv, lower_red, upper_red )

    # Adding two masks to generate the final mask
    mask = mask1 + mask2

    ## Open and Dilate the mask image
    mask = cv2.morphologyEx( mask, cv2.MORPH_DILATE, np.ones( (3, 3), np.uint8 ) )

    ##*********************************************************##
    #
    # 5. Generate final Augmented Output
    #
    ##*********************************************************##

    # Replacing pixels corresponding to cloak with the background pixels
    img[np.where( mask == 255 )] = background[np.where( mask == 255 )]
    out.write( img )
    cv2.imshow( 'Display', img )
    k = cv2.waitKey( 10 )
    if k == 27:
        break

# Close the window / Release webcam 
cap.release()
# After we release our webcam, we also release the output 
out.release()
# De-allocate any associated memory usage  
cv2.destroyAllWindows()