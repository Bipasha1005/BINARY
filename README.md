# BINARY

# Real-Time Violence Detection System

## Overview
The **Real-Time Violence Detection System** is an AI-powered web application designed to detect violent activities in real time. The system uses deep learning models to analyze video feeds and alert authorities when violence is detected. Additionally, it features live location tracking for enhanced situational awareness.

## Features
- **Real-time violence detection** using a trained deep learning model with our own built model using Conv3D, Maxpooling3D, Dense Layer etc.
- **Web-based user interface** for monitoring video feeds.
- **Live location tracking** for emergency response with alert sms to the contacts.
- **High accuracy** in detecting violent actions.

## Tech Stack
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Flask)
- **Machine Learning Model**: TensorFlow/Keras
- **Additional Tools**: OpenCV, W&B, YOLOv5

## Model Download
Download the trained deep learning model (`model.h5`) from the following link:
[Click here to download](https://drive.google.com/file/d/1a8ApzsWOuqXQyK5qzVxKIRb1Mhj55Brg/view?usp=drive_link)

## Installation
1. Clone the repository:
   ```bash
   https://github.com/Animeshghosh07/BINARY.git
   cd violence-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the `model.h5` file in the `models` directory.
4. Run the application:
   ```bash
   python app.py
   ```
5. Open your browser and navigate to `http://127.0.0.1:5000`.

## Usage
- Upload or stream a video feed.
- The system will analyze the frames in real time.
- If violence is detected, an alert will be triggered.
- Live location data will be sent to authorities (if enabled from the system).

## Contributors
- [Animesh Ghosh](https://github.com/Animeshghosh07)
- [Anurag Kumar Thakur](https://github.com/anuragthakur19)
- [Bipasha Acharjee](https://github.com/bipasha1005)

## Acknowledgments
- TensorFlow/Keras for deep learning
- OpenCV for image processing
- WebRTC for real-time video streaming


