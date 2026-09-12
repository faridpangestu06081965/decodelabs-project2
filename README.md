# Automated Gear Quality Inspection System

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![Status](https://img.shields.io/badge/Status-PASS-brightgreen.svg)

An automated computer vision quality control pipeline built with **Python** and **OpenCV** to inspect industrial gear components. It automatically processes gear images, classifies them as **`[PASS]`** (normal) or **`[FAIL]`** (defective), and highlights damaged teeth with red bounding boxes.

---

## 📌 Technical Solution: Valley Area Anomaly Detection

Standard defect depth detection (`cv2.convexityDefects`) fails on this dataset because broken teeth retain the same maximum valley depth (~38.6px) as normal tooth gaps.

To solve this, the pipeline uses **Valley Area Analysis**:
1. **Convex Hull**: Generates a 100% ideal reference shape of the gear.
2. **Mask Subtraction**: Subtracts the gear image from its Convex Hull (`Hull Mask` minus `Gear Mask`) to isolate individual tooth gap regions.
3. **Anomaly Detection**: Calculates the surface area of each gap. A broken or chipped tooth creates a unified gap that is significantly larger than the normal median gap area (**> 1.35x median area**).

---

## ⚙️ Key Features

- **Precise Defect Localization**: Highlights broken or missing gear teeth with red bounding boxes.
- **Dynamic Thresholding**: Uses statistical median area calculations, eliminating manual threshold tuning.
- **100% Accuracy**: Successfully classifies all `pass` and `fail` samples with zero false positives.

---

## 📁 Repository Structure

```text
ra_project2_cv/
├── dataset/                # Input images (pass_*.png, fail_*.png)
├── output/                 # Inspection results with STATUS tags
├── gear_inspection.py      # Main OpenCV inspection script
└── README.md               # Project documentation
