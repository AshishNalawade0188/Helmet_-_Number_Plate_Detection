import streamlit as st
import cv2
import numpy as np
import torch
import sqlite3
import os
from datetime import datetime
from ultralytics import YOLO
from PIL import Image

# Define new database path
DB_FOLDER = "database"
DB_PATH = os.path.join(DB_FOLDER, "detection_records.db")  # New database

# Ensure the database folder exists
os.makedirs(DB_FOLDER, exist_ok=True)

# Create and connect to the SQLite database
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# Create a unified table for all detections
cursor.execute("""
CREATE TABLE IF NOT EXISTS detection_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    vehicle_count INTEGER,
    person_count INTEGER,
    helmet_detected INTEGER
)
""")
conn.commit()

# Load the trained YOLO model
model_path = "best.pt"  # Ensure this file is in the same directory or provide the full path
model = YOLO(model_path)

# Define class names
class_names = ["Helmet", "Number Plate", "Person", "Motorbike"]

# Streamlit App Title & Styling
st.set_page_config(page_title="Helmet & Number Plate Detection", layout="wide")
st.title("🚀 Helmet & Number Plate Detection System")

# Create Tabs
tab1, tab2, tab3 = st.tabs(["📷 Image Detection", "📹 Video/Camera", "📊 Database"])

# ------------------------------ TAB 1: Image Detection ------------------------------ #
with tab1:
    st.header("📸 Upload an Image for Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        results = model(image)
        
        helmet_detected, vehicle_count, person_count = False, 0, 0
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box)
                label = class_names[int(cls)]
                color = (0, 255, 0)  # Default Green

                if label == "Person":
                    person_count += 1
                    color = (0, 255, 255)  # Yellow for Person
                elif label == "Helmet":
                    helmet_detected = True
                    color = (0, 255, 0)  # Green for Helmet
                elif label == "Number Plate" and not helmet_detected:
                    color = (0, 0, 255)  # Red if helmet is missing
                elif label == "Motorbike":
                    vehicle_count += 1
                    color = (255, 0, 0)  # Blue for Motorbike
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO detection_records (timestamp, vehicle_count, person_count, helmet_detected) VALUES (?, ?, ?, ?)",
                       (timestamp, vehicle_count, person_count, int(helmet_detected)))
        conn.commit()
        
        st.image(image, caption="Processed Image", channels="BGR", use_column_width=True)
        st.write(f"**🔹 Vehicles Detected:** `{vehicle_count}`")
        st.write(f"**🟢 Helmet Detected:** `{'Yes' if helmet_detected else 'No'}`")
        st.write(f"**👤 Persons Detected:** `{person_count}`")
        st.write(f"**🕒 Timestamp:** `{timestamp}`")

# ------------------------------ TAB 2: Video & Live Camera Detection ------------------------------ #
with tab2:
    st.header("📹 Video & Live Camera Detection")

    option = st.radio("Choose Input Type:", ("📂 Upload Video", "🎥 Use Webcam"))

    def process_video(cap):
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            helmet_detected, vehicle_count, person_count = False, 0, 0

            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                for box, cls in zip(boxes, classes):
                    x1, y1, x2, y2 = map(int, box)
                    label = class_names[int(cls)]
                    color = (0, 255, 0)

                    if label == "Person":
                        person_count += 1
                        color = (0, 255, 255)
                    elif label == "Helmet":
                        helmet_detected = True
                        color = (0, 255, 0)
                    elif label == "Number Plate" and not helmet_detected:
                        color = (0, 0, 255)
                    elif label == "Motorbike":
                        vehicle_count += 1
                        color = (255, 0, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("INSERT INTO detection_records (timestamp, vehicle_count, person_count, helmet_detected) VALUES (?, ?, ?, ?)",
                           (timestamp, vehicle_count, person_count, int(helmet_detected)))
            conn.commit()

            stframe.image(frame, channels="BGR", use_column_width=True)

        cap.release()

    if option == "📂 Upload Video":
        uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            temp_video_path = "temp_video.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.read())
            cap = cv2.VideoCapture(temp_video_path)
            process_video(cap)

    elif option == "🎥 Use Webcam":
        cap = cv2.VideoCapture(0)
        process_video(cap)

# ------------------------------ TAB 3: Database Records ------------------------------ #
with tab3:
    st.header("📊 Detection Records")
    cursor.execute("SELECT * FROM detection_records ORDER BY id DESC")
    rows = cursor.fetchall()
    if rows:
        st.subheader("📋 All Detection Records")
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info("No records found.")

conn.close()