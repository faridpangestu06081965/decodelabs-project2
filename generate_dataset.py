import cv2
import numpy as np
import os
import math

os.makedirs("dataset", exist_ok=True)

def create_gear(num_teeth=16, has_defect=False):
    img = np.zeros((400, 400), dtype=np.uint8)
    center = (200, 200)
    r_inner = 80
    r_outer = 120

    points = []
    for i in range(num_teeth):
        a1 = (2 * math.pi / num_teeth) * i
        a2 = a1 + (math.pi / num_teeth) * 0.35
        a3 = a1 + (math.pi / num_teeth) * 0.65
        a4 = (2 * math.pi / num_teeth) * (i + 1)

        points.append((center[0] + int(r_inner * math.cos(a1)), center[1] + int(r_inner * math.sin(a1))))
        points.append((center[0] + int(r_outer * math.cos(a2)), center[1] + int(r_outer * math.sin(a2))))
        points.append((center[0] + int(r_outer * math.cos(a3)), center[1] + int(r_outer * math.sin(a3))))
        points.append((center[0] + int(r_inner * math.cos(a4)), center[1] + int(r_inner * math.sin(a4))))

    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], 255)
    cv2.circle(img, center, 35, 0, -1)

    if has_defect:
        defect_angle = (2 * math.pi / num_teeth) * 4
        dx = int(center[0] + r_outer * math.cos(defect_angle))
        dy = int(center[1] + r_outer * math.sin(defect_angle))
        cv2.circle(img, (dx, dy), 22, 0, -1)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for i in range(1, 11):
    cv2.imwrite(f"dataset/pass_{i}.png", create_gear(has_defect=False))
    cv2.imwrite(f"dataset/fail_{i}.png", create_gear(has_defect=True))

print("✅ Sukses! 20 gambar dataset berhasil dibuat di folder 'dataset/'.")
