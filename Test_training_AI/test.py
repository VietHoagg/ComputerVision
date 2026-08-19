import cv2 as cv
img = cv.imread('b53ea1c8-7134-487e-90af-7c6c4bcb3863.jpg')
print(img)
cv.imshow('b53ea1c8-7134-487e-90af-7c6c4bcb3863.jpg', img)
cv.waitKey(0)
cv.destroyAllWindows()