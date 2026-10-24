# Real-Time Emotion Detection using Python, MediaPipe & DeepFace

A real-time facial emotion recognition system developed using **Python**, **OpenCV**, **MediaPipe**, and **DeepFace**. The application detects faces from a live webcam feed, extracts facial landmarks, predicts emotions, and displays a facial mesh with real-time emotion probability bars.

---

## Features

- Real-time webcam-based emotion detection
- Face detection and facial landmark extraction
- Face mesh visualization using MediaPipe
- Emotion prediction using DeepFace
- Live emotion probability bars
- FPS (Frames Per Second) monitoring
- Multi-threaded processing for improved performance
- Lightweight desktop application

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- DeepFace
- NumPy
- TensorFlow

---

## Project Structure

```
Emotion-Detection/

│── emotion.py
│── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/rohith2005-code/Emotion-Detection.git
```

### Navigate to the Project

```bash
cd Emotion-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Project

```bash
python emotion.py
```

---

## How It Works

1. Captures live video from the webcam.
2. Detects the face using MediaPipe.
3. Extracts facial landmarks.
4. Processes the detected face using DeepFace.
5. Predicts facial emotions.
6. Displays:

- Face Mesh
- Emotion Label
- Emotion Confidence
- Emotion Probability Bars
- FPS Counter

---

## Supported Emotions

- Happy
- Sad
- Angry
- Fear
- Surprise
- Neutral
- Disgust

---

## Applications

- Human Computer Interaction (HCI)
- Emotion Aware Systems
- Smart Surveillance
- Healthcare Monitoring
- Educational AI Projects
- Human-Robot Interaction
- AI Research

---

## Future Improvements

- Multiple face emotion detection
- Emotion history logging
- Attendance system integration
- Speech interaction
- Emotion-based chatbot integration
- Web application using Django

---



## Requirements

Example `requirements.txt`

```
opencv-python
mediapipe
deepface
numpy
tensorflow
```

---

## Author

**Rohith Rajesh**

B.Tech Artificial Intelligence & Machine Learning

GitHub: https://github.com/rohith2005-code

LinkedIn: https://www.linkedin.com/in/rohith-rajesh-ai

---

## License

This project is intended for educational and learning purposes.
