import os
import streamlit as st
import cv2
import dlib
import numpy as np
import time
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Face Detection Benchmark", layout="wide")
st.title("Face Detection Benchmark System")
st.write("Compare speed and accuracy of popular Face Detection algorithms.")

# 2. Model Loading Section (Simplified for Cloud)
@st.cache_resource
def load_models():
    # DNN Model (Caffe)
    proto = "deploy.prototxt"
    model = "res10_300x300_ssd_iter_140000.caffemodel"
    
    # تأكدي إن الملفات دي مرفوعة في نفس الفولدر على GitHub
    if not os.path.exists(proto) or not os.path.exists(model):
        st.error(f"Critical Error: Missing model files ({proto} or {model}) in repository.")
        return None, None, None
        
    net = cv2.dnn.readNetFromCaffe(proto, model)
    
    # Haar Cascade (OpenCV بيجيبها أوتوماتيك)
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Dlib HOG
    hog = dlib.get_frontal_face_detector()
    
    return net, haar, hog

net, haar, hog = load_models()

# Sidebar Configuration
st.sidebar.header("Experiment Settings")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None and net is not None:
    # Convert image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    col1, col2, col3 = st.columns(3)

    # --- 1. Haar Cascades ---
    with col1:
        st.subheader("Haar Cascades")
        start_time = time.time()
        faces_haar = haar.detectMultiScale(gray, 1.1, 4)
        end_time = time.time()
        
        res_haar = img_array.copy()
        for (x, y, w, h) in faces_haar:
            cv2.rectangle(res_haar, (x, y), (x+w, y+h), (255, 0, 0), 5)
        
        st.image(res_haar, use_container_width=True)
        st.metric("Inference Time", f"{(end_time - start_time)*1000:.2f} ms")
        st.write(f"Faces detected: {len(faces_haar)}")

    # --- 2. Dlib (HOG) ---
    with col2:
        st.subheader("Dlib (HOG)")
        start_time = time.time()
        faces_dlib = hog(gray)
        end_time = time.time()
        
        res_dlib = img_array.copy()
        for face in faces_dlib:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(res_dlib, (x, y), (x+w, y+h), (0, 255, 0), 5)
            
        st.image(res_dlib, use_container_width=True)
        st.metric("Inference Time", f"{(end_time - start_time)*1000:.2f} ms")
        st.write(f"Faces detected: {len(faces_dlib)}")

    # --- 3. DNN (SSD) ---
    with col3:
        st.subheader("Deep Learning (SSD)")
        start_time = time.time()
        (h_img, w_img) = img_cv.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img_cv, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        end_time = time.time()
        
        res_dnn = img_array.copy()
        dnn_count = 0
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w_img, h_img, w_img, h_img])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(res_dnn, (startX, startY), (endX, endY), (0, 0, 255), 5)
                dnn_count += 1
                
        st.image(res_dnn, use_container_width=True)
        st.metric("Inference Time", f"{(end_time - start_time)*1000:.2f} ms")
        st.write(f"Faces detected: {dnn_count}")

elif net is None:
    st.warning("Please make sure model files are uploaded to the repository.")
else:
    st.info("Please upload an image from the sidebar to start the benchmark.")
