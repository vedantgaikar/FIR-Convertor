import cv2
import numpy as np
from sklearn.decomposition import PCA

def extract_features(gray):
    """Extract fingerprint features using Harris corners and SIFT."""
    # Harris corner detection
    harris = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    harris = cv2.dilate(harris, None)

    # SIFT keypoints & descriptors
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors


def reduce_features(descriptors, n_components=32):
    """Reduce dimensionality of feature descriptors using PCA safely."""
    if descriptors is None or len(descriptors) == 0:
        return None

    # Ensure n_components is valid
    #max_components = min(descriptors.shape[0], descriptors.shape[1])
    #n_components = min(n_components, max_components)

    if n_components < 1:
        return descriptors  # not enough data for PCA, return original

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(descriptors)
    return reduced

