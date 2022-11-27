import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('3.png')
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height = grayImage.shape[0]
width = grayImage.shape[1]
r, thresh = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
k = np.ones((5, 5), dtype=np.uint8)
p = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k, iterations=2)
distTransform = cv2.distanceTransform(p, cv2.DIST_L2, 5)
r, f = cv2.threshold(distTransform, 0.2 * distTransform.max(), 255, 0)
b = cv2.dilate(p, k, iterations=3)
f = np.uint8(f)
r, m = cv2.connectedComponents(f)
unknown = cv2.subtract(b, f)
m= m + 1
m[unknown== 255] = 0
m = cv2.watershed(img, m)
result = img.copy()
result[m == -1] = [255, 0, 0]
plt.subplot(121), plt.imshow(grayImage,'gray')
plt.subplot(122), plt.imshow(result)
plt.show()
