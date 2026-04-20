import streamlit as st
import cv2
import numpy as np
import dlib
import time
from PIL import Image

st.set_page_config(page_title="Face Detection Benchmark", layout="wide")
st.title(" Face Detection Algorithm Comparison")
st.write("upload one photo and see the difference between two models now ")

@st.cache_resource
def load_models():
    # DNN SSD
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    # Haar Cascade
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # HOG
    hog = dlib.get_frontal_face_detector()
    return net, haar, hog

net, haar, hog = load_models()

uploaded_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    col1, col2, col3 = st.columns(3)

    # --- 1. Haar Cascade ---
    with col1:
        st.subheader("Haar Cascade")
        start = time.time()
        faces = haar.detectMultiScale(gray, 1.1, 5)
        t = (time.time() - start) * 1000
        
        canvas = img_rgb.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(canvas, (x, y), (x+w, y+h), (255, 0, 0), 3)
        
        st.image(canvas, use_column_width=True)
        st.metric("Time", f"{t:.2f} ms")
        st.write(f"Detected: {len(faces)}")

    # --- 2. HOG + SVM ---
    with col2:
        st.subheader("HOG + SVM")
        start = time.time()
        faces_hog = hog(gray)
        t = (time.time() - start) * 1000
        
        canvas = img_rgb.copy()
        for face in faces_hog:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
        st.image(canvas, use_column_width=True)
        st.metric("Time", f"{t:.2f} ms")
        st.write(f"Detected: {len(faces_hog)}")

    # --- 3. DNN SSD ---
    with col3:
        st.subheader("Deep Learning (SSD)")
        start = time.time()
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        
        canvas = img_rgb.copy()
        count = 0
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.5:
                count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        t = (time.time() - start) * 1000
        st.image(canvas, use_column_width=True)
        st.metric("Time", f"{t:.2f} ms")
        st.write(f"Detected: {count}")
