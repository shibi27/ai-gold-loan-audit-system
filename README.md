# 🤖 AI-Powered Gold Loan Audit System

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green?logo=opencv&logoColor=white)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet)
![License](https://img.shields.io/badge/License-MIT-success)
![Status](https://img.shields.io/badge/Build-Stable-brightgreen)

---

An intelligent desktop application built with **Python, OpenCV, and Tkinter** that automatically analyzes gold chain images to estimate **length, contour properties, and composition (gold vs stone percentage)**.  
The system also provides **AI-based image comparison** to verify chain authenticity.

---

## 🧠 Features

- 🟡 **Chain Analysis**
  - Detects gold chain contours using advanced edge and morphological filters.
  - Calculates total **chain length (in pixels)** using skeletonization.
  - Determines **gold and stone composition percentage** via HSV color analysis.
  - Displays a **pie chart** of composition inside the app.

- 🧩 **Image Comparison**
  - Compare two gold chain images using multiple algorithms (SIFT, ORB, AKAZE, etc.).
  - Computes **similarity score, confidence ratio, and SSIM index**.
  - Shows **best match visualization** with keypoints drawn.

- 🎨 **Modern Tkinter UI**
  - Tab-based layout (`Chain Analysis` / `Image Comparison`).
  - Embedded Matplotlib charts and image displays.
  - Status bar with real-time progress messages.

---

## 🛠️ Tech Stack

| Category | Tools |
|-----------|--------|
| Programming Language | Python 3.x |
| GUI Framework | Tkinter |
| Image Processing | OpenCV |
| Visualization | Matplotlib |
| Mathematical Analysis | NumPy, scikit-image |
| Data Handling | Pandas |

---

## 📦 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/shibi27/ai-gold-loan-audit-system.git
   cd ai-gold-loan-audit-system
