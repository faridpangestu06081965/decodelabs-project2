import cv2
import numpy as np
import glob
import os

os.makedirs("output", exist_ok=True)

def inspect_gear(image_path):
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

    # 1. Buat Mask Gear dan Mask Convex Hull (Bentuk Utuh Ideal)
    gear_mask = np.zeros_like(gray)
    cv2.drawContours(gear_mask, [gear_contour], -1, 255, -1)

    hull = cv2.convexHull(gear_contour)
    hull_mask = np.zeros_like(gray)
    cv2.drawContours(hull_mask, [hull], -1, 255, -1)

    # 2. Potong area celah/lembah gigi (Hull minus Gear)
    valleys_mask = cv2.subtract(hull_mask, gear_mask)

    # 3. Cari kontur dari setiap celah
    valley_contours, _ = cv2.findContours(valleys_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter noise kecil (< 20px)
    valid_valleys = [c for c in valley_contours if cv2.contourArea(c) > 20]

    if not valid_valleys:
        status = "PASS"
    else:
        areas = [cv2.contourArea(c) for c in valid_valleys]
        median_area = np.median(areas)
        
        is_fail = False
        # Celah yang ukurannya > 1.35x dari median celah normal menandakan gigi patah/sumbing
        for c in valid_valleys:
            area = cv2.contourArea(c)
            if area > 1.35 * median_area:
                is_fail = True
                x, y, w, h = cv2.boundingRect(c)
                # Anotasi kotak merah di lokasi gigi yang patah
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 10, y + h + 10), (0, 0, 255), 2)

        status = "FAIL" if is_fail else "PASS"

    color = (0, 0, 255) if status == "FAIL" else (0, 255, 0)
    cv2.putText(img, f"STATUS: [{status}]", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    filename = os.path.basename(image_path)
    cv2.imwrite(f"output/result_{filename}", img)
    print(f"File: {filename:<12} | Status: [{status}]")

print("--- AUTOMATED QUALITY INSPECTION (VALLEY AREA ANALYSIS) ---")
files = sorted(glob.glob("dataset/*.png"))
for f in files:
    inspect_gear(f)
