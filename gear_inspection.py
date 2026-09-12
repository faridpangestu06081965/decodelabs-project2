import cv2
import numpy as np
import glob
import os

os.makedirs("output", exist_ok=True)

def inspect_image(image_path, defect_threshold=50.0):
    img = cv2.imread(image_path)
    if img is None:
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return

    gear_contour = max(contours, key=cv2.contourArea)

    hull_indices = cv2.convexHull(gear_contour, returnPoints=False)
    defects = cv2.convexityDefects(gear_contour, hull_indices)

    is_defective = False
    
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i].reshape(-1)
            actual_distance = d / 256.0
            
            if actual_distance > defect_threshold:
                is_defective = True
                far_point = tuple(gear_contour[f][0])
                x, y = far_point
                cv2.rectangle(img, (x - 15, y - 15), (x + 15, y + 15), (0, 0, 255), 2)

    status = "FAIL" if is_defective else "PASS"
    color = (0, 0, 255) if is_defective else (0, 255, 0)
    cv2.putText(img, f"STATUS: [{status}]", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    filename = os.path.basename(image_path)
    cv2.imwrite(f"output/result_{filename}", img)
    print(f"File: {filename} | Status: [{status}] -> Saved to output/")

image_files = sorted(glob.glob("dataset/*.png"))
for file_path in image_files:
    inspect_image(file_path)
