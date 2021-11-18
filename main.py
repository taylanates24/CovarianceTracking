from covariance_tracker import CovarianceTracker
import cv2

bbox = [243, 67, 298, 218]
w = bbox[2] - bbox[0]
h = bbox[3] - bbox[1]
vid = cv2.VideoCapture("match5-c3.avi")
ret, frame = vid.read()

tracker = CovarianceTracker()
cov = tracker.covariance_matrix(frame, bbox=bbox)
cov = tracker.update(cov)

while True:

    ret2, frame2 = vid.read()
    distances = []
    coords = []
    for i in range(max(int(bbox[0] - (w / 2)), 0), min(int(bbox[2] - (w / 2)), frame2.shape[1] - w), 2):
        for j in range(max(int(bbox[1] - (h / 2)), 0), min(int(bbox[3] - (h / 2)), frame2.shape[0] - h), 2):
            bbox_ = [i, j, i + w, j + h]

            cov2 = tracker.covariance_matrix(frame2, bbox=bbox_)
            cov2 = tracker.update(cov2)

            dist = tracker.calc_distance(cov, cov2)
            distances.append(dist)
            coords.append(bbox_)

    bbox = coords[distances.index(min(distances))]
    image = cv2.rectangle(frame2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
    cv2.imshow("image", image)
    cv2.waitKey(25)

    if 0xFF == ord('q'):
        break
vid.release()

