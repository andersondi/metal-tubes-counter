import cv2
import numpy as np

kernel = np.ones((5, 5), np.uint8)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


imgOriginal = cv2.imread('resources/tubes3.jpg')

imgGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
imgGray = cv2.GaussianBlur(imgGray, (7, 7), cv2.BORDER_DEFAULT)

ret, imgThresholded = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)

morph = cv2.erode(imgThresholded, kernel, iterations=3)
morph = cv2.dilate(morph, kernel, iterations=3)

# morph = cv2.morphologyEx(imgThresholded, cv2.MORPH_CLOSE, kernel)

# imgTh2Gray = cv2.cvtColor(imgEroded, cv2.COLOR_GRAY2BGR)
# circles	= cv2.HoughCircles(imgTh2Gray,cv2.HOUGH_GRADIENT,1,40,param1=100,param2=30,minRadius=0,maxRadius=40)

circles	= cv2.HoughCircles(imgGray,cv2.HOUGH_GRADIENT,1,40,param1=100,param2=30,minRadius=0,maxRadius=40)

circles	= np.uint16(np.around(circles))

contours, hierarchy = cv2.findContours(morph,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

print (circles)
print (f"Encontrados {str(circles.shape[1])} circunferências")
print (f"Encontrados {str(len(contours))} regiões") 

count = 1
for	i in circles[0,:]:
 	# draw the outer circle
 	cv2.circle(imgOriginal,(i[0],i[1]),i[2],(0,255,0),4)
 	# draw the center of the circle
 	cv2.circle(imgOriginal,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow("Original image", imgOriginal)
cv2.imshow("Thresholded image", morph)
cv2.imshow("Blured gray image", imgGray)
cv2.waitKey(0)
cv2.destroyAllWindows()

