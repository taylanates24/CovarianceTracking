import cv2
import numpy as np



class CovarianceTracker:

    def __init__(self, bbox, vid):
        self.bbox = bbox
        self.vid = vid

    def feature_extraction(self, image, bbox):

        image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        lap = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
        lap = np.uint8(np.absolute(lap))
        lap_gray = cv2.cvtColor(lap, cv2.COLOR_BGR2GRAY)

        sobelx = cv2.Sobel(image, 0, dx=1, dy=0)
        sobelx = np.uint8(np.absolute(sobelx))
        sobelx_gray = cv2.cvtColor(sobelx, cv2.COLOR_BGR2GRAY)

        sobely = cv2.Sobel(image, 0, dx=0, dy=1)
        sobely = np.uint8(np.absolute(sobely))
        sobely_gray = cv2.cvtColor(sobely, cv2.COLOR_BGR2GRAY)

        return lap_gray, sobelx_gray, sobely_gray



