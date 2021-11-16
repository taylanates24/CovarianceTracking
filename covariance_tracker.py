import cv2
import numpy as np
from scipy.linalg import logm, expm, eigh


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

    def covariance_matrix(self, lap_gray, sobelx_gray, sobely_gray, bbox):
        lap_gray = lap_gray[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        sobelx_gray = sobelx_gray[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        sobely_gray = sobely_gray[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        y_len = bbox[3] - bbox[1]
        x_len = bbox[2] - bbox[0]

        x_coor_l = np.array(list(range(bbox[0], bbox[2])) * y_len).reshape(y_len, -1)
        y_coor_l = np.array((np.array(list(range(bbox[1], bbox[3]))).reshape(-1, 1).tolist() * x_len)).reshape(-1,
                                                                                                               y_len).transpose()

        f = np.concatenate(
            (y_coor_l[..., None], x_coor_l[..., None], lap_gray[..., None], sobelx_gray[..., None],
             sobely_gray[..., None]),
            axis=2).reshape(-1, 1, 5)

        f_mean = np.array(
            [f[:, :, 0].mean(), f[:, :, 1].mean(), f[:, :, 2].mean(), f[:, :, 3].mean(), f[:, :, 4].mean()])

        asd = f - f_mean
        total = np.dot(asd.transpose().squeeze(), asd.squeeze())

        total = total / len(f)

        return total

    def calc_distance(self, cov1, cov2):
        eigvals = eigh(cov1, cov2, eigvals_only=True)
        eig_ln = np.square(np.log(eigvals)).sum()

        return eig_ln


