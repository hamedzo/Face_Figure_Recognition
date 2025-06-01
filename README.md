# Face Pose & Body Language Detection

This repository provides tools to gather facial and pose landmarks using MediaPipe and OpenCV, train a machine learning model on the gathered data, and use the trained model for real-time body language classification.

---

## 📸 Project Overview

This project leverages **MediaPipe Holistic** (including Face Mesh, Pose, and Hands) to collect, analyze, and classify human pose and facial features in real time. Applications include:

- Facial landmark & pose detection
- Gesture and emotion recognition
- Head pose and body language estimation
- Baseline biometric/identity recognition

---

## 🎯 Objectives

- Detect human faces and poses from live video (webcam)
- Identify and annotate 468 facial landmarks per face and body pose
- Analyze facial and body metrics for emotion/gesture recognition
- Provide a workflow for gathering data, training, and real-time prediction

---

## ✨ Features

- 🧠 **Landmark Detection:** 468-point facial mesh, full body pose, and hand tracking using MediaPipe Holistic
- ⏱ **Real-Time Processing:** Smooth video stream handling via OpenCV
- 🧑‍🔬 **Data Collection:** Record landmarks with class labels for custom training
- 🤖 **Machine Learning:** Train, evaluate, and export classifiers (Random Forest, Logistic Regression, etc.) for pose/body language
- 🏷 **Live Prediction:** Run real-time predictions with overlayed results
- 💡 **Extensible Design:** Easily add new classes or features (emotion types, gestures, etc.)

---

## 🛠 Tech Stack

- [Python 3.7+](https://www.python.org/)
- [MediaPipe](https://mediapipe.dev/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)

---

## 📂 Project Structure

```bash
Face_Pose_Detection/
│
├── Data gathering.py       # Collect pose/face landmarks and save to CSV
├── learn model.py          # Train model on gathered data and export as pickle
├── load model.py           # Real-time prediction using webcam and trained model
├── body_language.pkl       # Example pre-trained model
├── README.md               # Project documentation
└── coords.csv              # Data file (generated during data collection)
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/hamedzo/Face_Pose_Detection.git
cd Face_Pose_Detection
```

### 2. (Optional) Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install mediapipe opencv-python numpy pandas scikit-learn
```

### 4. Data Gathering

Run the following script to collect pose and face landmarks from your webcam and save them to `coords.csv`:

```bash
python "Data gathering.py"
```

- Press `q` to quit.
- The script writes pose and face landmarks, along with a class label (default: `Happy`), to `coords.csv`.
- Change the `class_name` variable in the script to collect data for different classes.

### 5. Model Training

After gathering data, train a model:

```bash
python "learn model.py"
```

- Reads `coords.csv`, splits data, trains multiple classifiers, reports accuracy, and saves the Random Forest model as `body_language.pkl`.

### 6. Real-Time Prediction

Use your webcam for real-time classification:

```bash
python "load model.py"
```

- Loads the trained model, starts webcam, and overlays prediction results (class and confidence) on the video feed.

---

## 🧠 How It Works

1. **Data Collection:** Gather pose and face landmarks labeled with class names (e.g., happy, sad, surprised).
2. **Model Training:** Train classifiers to recognize the classes from landmark data, save the best model.
3. **Live Prediction:** Use the webcam and model to recognize and display predicted body language/emotion in real time.

---

## 📸 Sample Output

_Add sample images or GIFs here to showcase annotated video frames with predicted class and probability._

---

## ✅ Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- pandas
- scikit-learn

---

## 🧪 Example Use Cases

- 🎭 Real-time emotion/body language recognition
- 📊 Automated gesture analysis
- 🔐 Biometric authentication or mood detection
- 🧍 Avatar animation from facial/body gestures
- 📏 Scientific or biometric face/pose measurements

---

## 📌 Roadmap / To-Do

- [ ] Add more expression/body language categories
- [ ] Integrate 3D face/pose visualization
- [ ] Add GUI support (Tkinter, PyQt)
- [ ] Export landmark data as JSON
- [ ] Improve data balancing and augmentation

---

## 🤝 Contributing

Contributions are warmly welcomed! To contribute:

1. **Fork** the repository
2. **Create a branch:** `git checkout -b feature-name`
3. **Make your changes**
4. **Commit:** `git commit -m 'Add feature'`
5. **Push:** `git push origin feature-name`
6. **Open a Pull Request**

---

## 🧾 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

- **Author:** Hamed Zohrab
- **GitHub:** [@hamedzo](https://github.com/hamedzo)
- **Email:** hamed.zohrab@gmail.com

---

## 🙏 Acknowledgements

- [Google MediaPipe](https://mediapipe.dev/) for real-time ML pipelines
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [scikit-learn](https://scikit-learn.org/) for machine learning tools
- The open-source community for invaluable contributions

---
