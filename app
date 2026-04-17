import cv2
import numpy as np
import time
import pandas as pd
import dlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from google.colab.patches import cv2_imshow

!wget -N https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
!wget -N https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

print("--- Loading LFW Dataset ---")
lfw_people = fetch_lfw_people(min_faces_per_person=0, resize=None, color=True)
images = lfw_people.images
print(f" Successfully loaded {len(images)} images.")

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
hog_detector = dlib.get_frontal_face_detector()
dnn_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")



def run_benchmark(image_array, limit=1000):
    results_data = []
    print(f" Starting benchmark on {limit} images...")

    for i in range(limit):
        img_raw = image_array[i]
        img_uint8 = (img_raw * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        row = {'ID': i}

        start = time.time()
        faces_haar = haar_cascade.detectMultiScale(gray, 1.1, 5)
        row['Haar_Time_ms'] = (time.time() - start) * 1000
        row['Haar_Count'] = len(faces_haar)

        start = time.time()
        faces_hog = hog_detector(gray)
        row['HOG_Time_ms'] = (time.time() - start) * 1000
        row['HOG_Count'] = len(faces_hog)

        start = time.time()
        h, w = img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        dnn_net.setInput(blob)
        detections = dnn_net.forward()
        dnn_count = sum(1 for j in range(detections.shape[2]) if detections[0, 0, j, 2] > 0.5)
        row['DNN_Time_ms'] = (time.time() - start) * 1000
        row['DNN_Count'] = dnn_count

        results_data.append(row)
        if (i+1) % 200 == 0: print(f" Processed {i+1} images...")

    return pd.DataFrame(results_data)

df_results = run_benchmark(images, limit=1000)

def calculate_final_metrics(df, col):
    total = len(df)
    tp = (df[col] >= 1).sum()
    fp = (df[col] > 1).sum()
    fn = (df[col] == 0).sum()

    det_accuracy = (tp / total) * 100
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return det_accuracy, f1 * 100

summary = []
for algo in ['Haar', 'HOG', 'DNN']:
    acc, f1 = calculate_final_metrics(df_results, f'{algo}_Count')
    summary.append({
        'Algorithm': algo,
        'Avg Time (ms)': df_results[f'{algo}_Time_ms'].mean(),
        'Detection Accuracy (%)': acc,
        'F1-Score (%)': f1
    })

summary_df = pd.DataFrame(summary)
print("\n--- PERFORMANCE SUMMARY (1000 Images) ---")
print(summary_df.to_string(index=False))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

summary_df.plot(x='Algorithm', y='Avg Time (ms)', kind='bar', ax=ax1, color='salmon', title='Processing Speed (Lower is better)')
summary_df.plot(x='Algorithm', y='Detection Accuracy (%)', kind='bar', ax=ax2, color='skyblue', title='Detection Accuracy (Higher is better)')

plt.tight_layout()
plt.show()

test_idx = np.random.randint(0, 1000)
sample_raw = (images[test_idx] * 255).astype(np.uint8)
sample_bgr = cv2.cvtColor(sample_raw, cv2.COLOR_RGB2BGR)

# Example: Draw Haar Cascade in Red
faces_haar = haar_cascade.detectMultiScale(cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2GRAY), 1.1, 5)
for (x, y, w, h) in faces_haar:
    cv2.rectangle(sample_bgr, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(sample_bgr, 'Haar', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

print(f"\nSample Visualization (Index {test_idx}):")
cv2_imshow(sample_bgr)
streamlit
opencv-python-headless
dlib
pandas
numpy
Pillow
requests
