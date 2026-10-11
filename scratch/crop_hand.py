import cv2
import numpy as np

img = cv2.imread("ref image.jpeg")
print("Image shape:", img.shape)
# Top left hand is approximately [100:300, 30:200]
h, w = img.shape[:2]
crop_a = img[int(h*0.15):int(h*0.35), int(w*0.05):int(w*0.3)]
cv2.imwrite("scratch/crop_a.jpg", crop_a)
print("Saved crop_a.jpg with shape:", crop_a.shape)
