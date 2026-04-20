import streamlit as st
import cv2
import numpy as np
import dlib
import time
import os
import urllib.request

# --- 1. التحميل التلقائي للموديلات (أضمن طريقة) ---
def download_models():
    base_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    weights_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    if not os.path.exists("deploy.prototxt"):
        urllib.request.urlretrieve(base_url, "deploy.prototxt")
    if not os.path.exists("res10_300x300_ssd_iter_140000.caffemodel"):
        urllib.request.urlretrieve(weights_url, "res10_300x300_ssd_iter_140000.caffemodel")

# --- 2. إعداد الصفحة ---
st.set_page_config(page_title="Face Detection Benchmark", layout="wide")
st.title("👤 Face Detection Algorithm Comparison")

# --- 3. تحميل الموديلات مع معالجة الأخطاء ---
@st.cache_resource
def load_models():
    download_models()
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    hog = dlib.get_frontal_face_detector()
    return net, haar, hog

try:
    net, haar, hog = load_models()
    st.success("Models loaded successfully! ✅")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop() # بيوقف الكود هنا لو فيه مشكلة عشان ميجيبش spawn error

# --- 4. باقي كود رفع الصور (نفس الكود اللي معاكي) ---
uploaded_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # كملي باقي الكود هنا...
    st.write("Image uploaded!")
