import cv2
import numpy as np
from sklearn.decomposition import PCA

def extract_features(gray):
    harris = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    harris = cv2.dilate(harris, None)

    #sift
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors


def reduce_features(descriptors, n_components=32):

    if descriptors is None or len(descriptors) == 0:
        return None

    if n_components < 1:
        return descriptors 

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(descriptors)
    return reduced

