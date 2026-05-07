# Sleep-Detection-
Real-time drowsiness detection using MediaPipe &amp; Eye Aspect Ratio
#  Sleep Detection System

Real-time drowsiness detection using MediaPipe Face Mesh & Eye Aspect Ratio (EAR). Monitors eye closure duration and triggers audio alerts when sleep is detected.

## 🚀 Features
- 👁️ Real-time eye tracking using MediaPipe (468 facial landmarks)
- 📐 Eye Aspect Ratio (EAR) algorithm for precise blink/closure detection
- 🔊 Audio alert system when prolonged eye closure is detected
-  Tkinter GUI with live video feed & status dashboard
- 🎚️ Live adjustable thresholds (EAR sensitivity & alert delay)
-  Tracks blink count, alerts triggered, and session duration
- 🔇 Mute toggle for quiet environments

## 🛠️ Tech Stack
- Python 3.x
- OpenCV (`cv2`)
- MediaPipe
- NumPy
- Tkinter & Pillow (GUI)
- `winsound` (Windows audio)

## 📦 Installation
1. Download or clone this repository
2. Install required packages:
   ```bash
   pip install opencv-python mediapipe numpy Pillow
