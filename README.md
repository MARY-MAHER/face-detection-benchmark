#  Face Detection Benchmark System

An interactive web application built with **Streamlit** to compare different face detection algorithms. This project demonstrates the trade-offs between speed and accuracy across various Computer Vision techniques.

##  Live Demo
[Put your Streamlit Cloud Link Here]

##  Features
* **Multi-Model Comparison:** Compare three popular detection methods:
  * **Haar Cascades:** Fast, lightweight, classic OpenCV approach.
  * **HOG (Histogram of Oriented Gradients):** Robust detection using the Dlib library.
  * **DNN (Deep Neural Networks):** High-accuracy SSD model for complex environments.
* **Performance Metrics:** Real-time tracking of processing time (in milliseconds) for each algorithm.
* **Interactive UI:** Dynamic image uploader and visual overlays for detection results.
* **Auto-Deployment Ready:** Cloud-optimized with automated model weight handling.

##  Tech Stack
* **Language:** Python
* **Libraries:** Streamlit, OpenCV, Dlib, NumPy, PIL.
* **Deployment:** Streamlit Cloud.

##  Project Structure
* `app.py`: The main Streamlit application logic.
* `requirements.txt`: List of dependencies for cloud environment setup.
* `README.md`: Project documentation.

##  Installation & Local Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/face-detection-benchmark.git](https://github.com/YourUsername/face-detection-benchmark.git)
