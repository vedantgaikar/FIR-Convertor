import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os

from preprocessing import preprocess_image
from feature_extraction import extract_features, reduce_features
from fir_utils import save_fir, load_fir, match_fingerprints

# ---------------- GUI Functions ----------------

def convert_to_fir():
    """Convert a BMP fingerprint image to FIR file and save keypoints visualization."""
    file_path = filedialog.askopenfilename(filetypes=[("BMP files", "*.bmp")])
    if not file_path:
        return

    # Step 1: Preprocessing
    img, gray, binary = preprocess_image(file_path)

    # Step 2: Feature Extraction
    keypoints, descriptors = extract_features(gray)

    if descriptors is None:
        messagebox.showerror("Error", "No features detected in the image.")
        return

    # Step 3: Dimensionality Reduction (PCA)
    reduced = reduce_features(descriptors)
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Define output folder
    output_dir = r"D:\Codes\FIR Convertor\output"
    keypoint_dir = r"D:\Codes\FIR Convertor\keypoints"

    os.makedirs(output_dir, exist_ok=True)

    # Step 4: Save FIR file with same name as input
    fir_path = os.path.join(output_dir, base_name + ".fir")
    save_fir(fir_path, reduced)

    # Step 5: Save image with keypoints drawn (same name + _keypoints.png)
    kp_img = cv2.drawKeypoints(
        img, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    out_img_path = os.path.join(keypoint_dir, base_name + "_keypoints.png")
    cv2.imwrite(out_img_path, kp_img)



    messagebox.showinfo(
        "Conversion Successful",
        f"FIR file saved:\n{fir_path}\n\nKeypoints image saved:\n{out_img_path}"
    )


def match_two_fingerprints():
    """Match two FIR files and show similarity score."""
    file1 = filedialog.askopenfilename(title="Select First FIR", filetypes=[("FIR files", "*.fir")])
    file2 = filedialog.askopenfilename(title="Select Second FIR", filetypes=[("FIR files", "*.fir")])

    if not file1 or not file2:
        return

    desc1 = load_fir(file1)
    desc2 = load_fir(file2)

    if desc1 is None or desc2 is None:
        messagebox.showerror("Error", "One of the FIR files is invalid or empty.")
        return

    matches, score = match_fingerprints(desc1, desc2)
    messagebox.showinfo("Match Result", f"Fingerprint Match Score: {score}\nTotal Matches: {len(matches)}")


# ---------------- Tkinter UI ----------------
def main():
    root = tk.Tk()
    root.title("Fingerprint Feature Extraction System")
    root.geometry("400x200")

    tk.Label(root, text="Fingerprint Processing System", font=("Arial", 14, "bold")).pack(pady=10)

    btn1 = tk.Button(root, text="Convert BMP to FIR", command=convert_to_fir, width=30)
    btn1.pack(pady=10)

    btn2 = tk.Button(root, text="Match Two Fingerprints", command=match_two_fingerprints, width=30)
    btn2.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
