# 🛡️ AI Proctoring System

A smart, AI-powered solution for secure and fair online examinations. This system uses real-time computer vision and audio processing techniques to detect and prevent cheating by monitoring for unauthorized presence, gadget usage, unusual head movements, lip activity, and suspicious audio input.

---

## 📌 Features

### 1. 👤 Person Detection
- Uses **YOLOv3** and **SSD** to detect the number of individuals.
- Alerts and terminates exam if:
  - No person is detected.
  - More than one person is detected.

### 2. 📱 Gadget Detection
- Detects mobile phones or electronic devices using **YOLOv3** with **COCO Dataset**.
- Takes real-time snapshots every 300ms.
- Triggers alerts and ends the test if a gadget is detected.

### 3. 🧠 Head-Pose Tracking
- Utilizes **YOLOv3** with **OpenCV PnP** model to track head movements and gaze direction.
- Flags abnormal behavior such as looking away from the screen for extended periods.
- Auto-terminates the session if suspicious behavior is detected.

### 4. 👄 Lip Movement Tracking
- Tracks excessive or continuous lip movement, which may indicate verbal cheating.
- Integrates **speech recognition** with lip detection to increase accuracy.
- Triggers an alert and ends the test on continuous mouth activity.

### 5. 🎙️ Audio Monitoring
- Uses **Google Speech Recognition API** and **NumPy** for real-time microphone analysis.
- Detects unauthorized speaking not related to exam content.
- Issues warning and ends test if threshold is exceeded.

---

## 🧠 Technologies Used

- **YOLOv3** – Object detection
- **OpenCV** – Computer vision processing
- **CNN / Deep Learning** – Behavior analysis
- **Google Speech Recognition API** – Audio input analysis
- **Tkinter** – Alert system GUI
- **NumPy** – Audio analysis
- **Python** – Backend development

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/AI-Proctor.git
cd AI-Proctor
pip install -r requirements.txt
