import pickle
import cv2

def save_fir(file_path, descriptors):
    with open(file_path, "wb") as f:
        pickle.dump(descriptors, f)


def load_fir(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except:
        return None

def match_fingerprints(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)

    if not matches:
        return [], 0.0

    #less distance more match
    distances = [m.distance for m in matches]

    avg_distance = sum(distances) / len(distances)
    score = 100 / (1 + avg_distance)   #more score is more match

    return matches, score


