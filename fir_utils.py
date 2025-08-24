import pickle
import cv2

def save_fir(file_path, descriptors):
    """Save descriptors to FIR file (pickle format)."""
    with open(file_path, "wb") as f:
        pickle.dump(descriptors, f)


def load_fir(file_path):
    """Load descriptors from FIR file."""
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except:
        return None

def match_fingerprints(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)

    if not matches:
        return [], 0.0  # No matches found

    # Smaller distance = better match
    distances = [m.distance for m in matches]

    # Normalize score (inverse of average distance)
    avg_distance = sum(distances) / len(distances)
    score = 100 / (1 + avg_distance)   # Higher score = better match

    return matches, score


