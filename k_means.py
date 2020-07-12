import numpy as np
import cv2
import matplotlib.pyplot as plt


img = cv2.imread("butterfly.jpeg")
img_copy= np.copy(img)
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
#plt.imshow(img_copy)


#Rehape to 2D array of pixel values and 3 color values
pixel_val = img_copy.reshape((-1, 3))

#convert to float type
pixel_val = np.float32(pixel_val)

#Now perform k means
k =8;

#decide criteria to converg
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0  )

retval, labels, centers = cv2.kmeans(pixel_val, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

#Display the image with labels we get

#convert data to 8-bit values
centers = np.uint(centers)
segmented_data = centers[labels.flatten()]

#Reshape data to origimal img dimentions
segmented_img = segmented_data.reshape((img_copy.shape))
labels_reshape = labels.reshape(img_copy.shape[0], img_copy.shape[1])

plt.imshow(segmented_img)