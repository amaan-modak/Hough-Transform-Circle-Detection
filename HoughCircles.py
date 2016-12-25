import cv2,sys
import numpy as np
import time
import operator
import math

np.seterr(over='ignore')

# Function defined to display the image
def display(title,source):
  cv2.imshow(title,source)
  cv2.waitKey(1)

# Function defined to draw circles
def output_circles(image,x0,y0,r,color,thickness):
  for t in xrange(0,360):
    x=int(x0+r*(math.cos(math.radians(t))))
    y=int(y0+r*(math.sin(math.radians(t))))
    for d in range(thickness):
      image.itemset((y,x,0),color[0])
      image.itemset((y,x,1),color[1])
      image.itemset((y,x,2),color[2])
  display("Circled Image",image)

start_time = time.time() # Start time of Algorithm Implementation

original_image = cv2.imread('Test_Image.jpg',1) # Input file image
display("Original Image",original_image)

output = original_image.copy()

#Gaussian Blurring of Image
blur_image = cv2.GaussianBlur(original_image,(3,3),0)
display("Blurred Image",blur_image)

# Canny Edge Detection
edged_image = cv2.Canny(blur_image,80,200)
display("Edge Detected Image",edged_image)

'''
#Using OpenCV Sobel Edge Detector to detect edges
sobelx = cv2.Sobel(blur_gray_image,cv2.CV_64F,1,0,ksize=3)  # x
sobely = cv2.Sobel(blur_gray_image,cv2.CV_64F,0,1,ksize=3)  # y
Sobel = np.hypot(sobelx,sobely)
Sobel = np.array(Sobel,dtype="float32")
Sobel *= 255.0 / np.max(Sobel)
ret,Sobel_edged_image = cv2.threshold(Sobel,55,255,cv2.THRESH_BINARY)
display("Sobel Edged Image",Sobel_edged_image)
'''

# Canny Edge Detector provides with a better edge detection output so we consider that output as the input for the Hough Transform Algorithm

# Hough Circle Implementation Starts Here

# Height and width of image
height = edged_image.shape[0]
width = edged_image.shape[1]

Rmin = 20 # Minimum Radius
Rmax = 60 # Maximum Radius

# General formula of a circle at center x0,y0
# x=x0+r*cos(t)
# y=y0+r*sin(t)

# Initialise Accumulator as a Dictionary with x0, y0 and r as tuples and votes as values
accumulator = {}

# Loop over the image
for y in xrange(0,height):
  for x in xrange(0,width):
    # If an edge pixel is found..
    if edged_image.item(y,x) == 255:
      # Loop over all the values of radius. Considering only even values for for minimizing performance time by setting increment as 2
      for r in xrange(Rmin,Rmax,2):
        # Looping over the values of theta, Considering only even values for for minimizing performance time by setting increment as 2
        for t in xrange(0,360,2):

          # Determining all the possible centres x0,y0 using the above formula
          x0 = int(x-(r*math.cos(math.radians(t))))
          y0 = int(y-(r*math.sin(math.radians(t))))

          # Checking if the center is within the range of image
          if x0>0 and x0<width and y0>0 and y0<height:
            # Voting process...
            if (x0,y0,r) in accumulator:
              accumulator[(x0,y0,r)]=accumulator[(x0,y0,r)]+1
            else:
              accumulator[(x0,y0,r)]=0

# Print out percentage of rows completed..
  sys.stdout.write("\r" + str(int(float(y+1)/float(height)*100))+"%")
  sys.stdout.flush()

end_time = time.time() # End time of Algorithm Implementation
print

thresh = 40  # Minimum votes required to be considered a circle in output
print "Default threshold =",thresh," (-1 to exit)"

entire_sorted_accumulator = sorted(accumulator.items(),key=operator.itemgetter(1),reverse=True) # Sort the accumulator in descending order of number of votes

while True:
  output = original_image.copy() # New output image for each threshold value from user input
  sorted_accumulator = entire_sorted_accumulator[:thresh]
  draw_circles = [] # This is used to store the circles after thresholding and rounding
  redundant_circles = {} # This is used to store the redundant circles based on rounding

  # roundr is responsible for minimum distance between two circles
  roundr = 15  # Round off radius. To ignore circles with similar radii with same center.
  roundc = 35  # Round off center. To ignore circles with nearby center with similar radius
  for circle in sorted_accumulator:
    if (circle[0][0]/roundc*roundc,circle[0][1]/roundc*roundc) in redundant_circles and redundant_circles[(circle[0][0]/roundc*roundc,circle[0][1]/roundc*roundc)]==circle[0][2]/roundr*roundr: # Checking if the circle obtained from accumulator is a redundant circle
      # print "Skipped",circle
      pass
    else:
      # print circle
      redundant_circles[(circle[0][0]/roundc*roundc,circle[0][1]/roundc*roundc)]=circle[0][2]/roundr*roundr # Add new circle to redundant circle
      draw_circles.append((circle[0][0],circle[0][1],circle[0][2]))

  for circle in draw_circles:
    output_circles(output,circle[0],circle[1],circle[2],(0,255,0),3) # Draw detected circles using the function defined above
  # cv2.waitKey(0)
  cv2.imwrite("Circled Image with Threshold Value "+str(thresh)+".png",output)

  # Hough Transform implementation with default threshold ends here

  # Code needs work to determine optimum threshold value directly from image dynamically (Adaptive)

  thresh = input("Threshold : ") # Taking threshold value as user input for trial and error method
  if thresh < 0:
    break
  cv2.destroyAllWindows()

time_taken = end_time - start_time # Total time taken starting from taking image as input to performing Hough Transform with default threshold
print 'Time taken for execution',time_taken

cv2.destroyAllWindows()