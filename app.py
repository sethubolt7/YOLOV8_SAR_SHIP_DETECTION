import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile

# Load the trained YOLO model
model = YOLO('weights/best.pt')

# Streamlit app title
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>YOLOv8 SAR Ship Detection</h1>
""", unsafe_allow_html=True)

# Upload options
option = st.radio("Select input type:", ('Image', 'Video'))

if option == 'Image':
    st.markdown("""
        <h2 style='text-align: left; color: white;'>Upload an Image for Ship Detection</h2>
    """, unsafe_allow_html=True)
    
    uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)

        # Perform inference
        results = model(image_np)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        object_count = len(boxes)

        # Count display
        st.markdown(f"""
            <h2 style='text-align: center; color: #FF5722;'>Ships Detected: {object_count}</h2>
        """, unsafe_allow_html=True)

        # Bounding-box-only image
        image_with_boxes = image_np.copy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (255, 165, 0), 2)  # Orange

        # Full detection image
        detected_image = results[0].plot()

        # Side-by-side layout
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_with_boxes, caption="Bounding Boxes Only", use_column_width=True)
        with col2:
            st.image(detected_image, caption="Detections with Labels", use_column_width=True)

elif option == 'Video':
    st.markdown("""
        <h2 style='text-align: left; color: white;'>Upload a Video for Ship Detection</h2>
    """, unsafe_allow_html=True)

    uploaded_video = st.file_uploader("", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)

        count_placeholder = st.empty()
        col1, col2 = st.columns(2)
        stframe_boxes_only = col1.empty()
        stframe_full = col2.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference
            results = model(frame)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            object_count = len(boxes)

            # Bounding-box-only frame
            frame_boxes_only = frame.copy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame_boxes_only, (x1, y1), (x2, y2), (255, 165, 0), 2)  # Orange

            # Full detection
            frame_full = results[0].plot()

            # Convert to RGB
            frame_boxes_only = cv2.cvtColor(frame_boxes_only, cv2.COLOR_BGR2RGB)
            frame_full = cv2.cvtColor(frame_full, cv2.COLOR_BGR2RGB)

            # Display count first
            count_placeholder.markdown(f"""
                <h2 style='text-align: center; color: #FF5722;'>Ships Detected: {object_count}</h2>
            """, unsafe_allow_html=True)

            # Side-by-side frames
            stframe_boxes_only.image(frame_boxes_only, caption="Bounding Boxes Only", channels="RGB", use_column_width=True)
            stframe_full.image(frame_full, caption="Detections with Labels", channels="RGB", use_column_width=True)

        cap.release()
        cv2.destroyAllWindows()
