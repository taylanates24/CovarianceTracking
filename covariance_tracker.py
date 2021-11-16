import cv2
import numpy as np


class CovarianceTracker:

    def __init__(self, bbox, vid):
        self.bbox = bbox
        self.vid = vid
