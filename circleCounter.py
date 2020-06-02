import cv2
import numpy as np

kernel = np.ones((5, 5), np.uint8)

# img = cv2.imread('circles1.png')
img = cv2.imread('tubes1.jpg')

imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgGray = cv2.GaussianBlur(imgGray, (7, 7), cv2.BORDER_DEFAULT)

# imgCanny = cv2.Canny(img, 50, 200)
# imgDialation = cv2.dilate(imgCanny, kernel, iterations = 1)
ret, imgThresholded = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)
# imgEroded = cv2.erode(imgThresholded, kernel, iterations = 1)

# cnts, hierarchy = cv2.findContours(imgThresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(source_image, cnts, -1, (255, 0, 0), 2)
# circles	= cv2.HoughCircles(thresholded_image_gray,cv2.HOUGH_GRADIENT,1,30,param1=50,param2=30,minRadius=0,maxRadius=100)
circles	= cv2.HoughCircles(imgGray,cv2.HOUGH_GRADIENT,1,40,param1=100,param2=30,minRadius=0,maxRadius=40)

circles	= np.uint16(np.around(circles))

print (circles)

print ("Encontrados " + str(circles.shape[1]) + " tubos")

count = 1
for	i in circles[0,:]:
	# draw the outer circle
	cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),4)
	# draw the center of the circle
	cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)


cv2.imshow("Original image", img)
cv2.imshow("HSV image", imgHsv)
cv2.imshow("Thresholded image", imgThresholded)
cv2.imshow("Blured gray image", imgGray)
# cv2.imshow("Canny image", imgCanny)
# cv2.imshow("Dialation image", imgDialation)
# cv2.imshow("Eroded image", imgEroded)
cv2.waitKey(0)
cv2.destroyAllWindows()
