import cv2

def preprocess_image(img_path):
    """Preprocess fingerprint: grayscale, Gaussian blur, histogram equalization, binary image."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Invalid image path or format.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian filter for noise removal
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Histogram Equalization for contrast enhancement
    eq = cv2.equalizeHist(blur)

    # Binary thresholding
    _, binary = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img, gray, binary
