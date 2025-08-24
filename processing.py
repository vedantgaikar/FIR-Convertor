import cv2
import numpy as np
import pickle
import os

def preprocess_image(image_path):
    """Read and preprocess the fingerprint image (grayscale + denoise)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def extract_sift_features(image):
    """Extract SIFT keypoints and descriptors."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def save_keypoints_image(image, keypoints, output_path):
    """Draw keypoints on the image and save."""
    img_kp = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(output_path, img_kp)

def save_fir_file(descriptors, fir_path):
    """Save descriptors as a .fir file using pickle."""
    with open(fir_path, 'wb') as f:
        pickle.dump(descriptors, f)

def process_fingerprint(image_path, output_dir="outputs"):
    """Full pipeline: preprocess → SIFT → visualize → FIR save."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = preprocess_image(image_path)
    keypoints, descriptors = extract_sift_features(img)

    # Save image with keypoints
    keypoint_image_path = os.path.join(output_dir, "fingerprint_keypoints.jpg")
    save_keypoints_image(img, keypoints, keypoint_image_path)

    # Save FIR file
    fir_file_path = os.path.join(output_dir, "fingerprint_data.fir")
    save_fir_file(descriptors, fir_file_path)

    return keypoint_image_path, fir_file_path, len(keypoints)
