import streamlit as st
import torch
import cv2
import shutil
from pathlib import Path
import os
import tempfile
from model_ex import main




# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/amazon_product_search_final/weights/yolov5x.pt')
names = yolo_model.names


# Streamlit App
st.title("Video Upload and YOLOv5 Object Detection")

# Upload video
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.video(video_path)
    # Define output directories
    output_dir = 'result_yolo_video/'
    output_video_path = os.path.join(output_dir, 'output_video.mp4')
    detected_objects_dir = os.path.join(output_dir, 'detected_objects')
    shutil.rmtree(detected_objects_dir, ignore_errors=True)
    os.makedirs(detected_objects_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Video processing
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = yolo_model(frame)

        boxes = results.xyxy[0].numpy()
        scores = results.xyxy[0][:, 4].numpy()
        class_ids = results.xyxy[0][:, 5].numpy()

        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            if score > 0.7:
                x1, y1, x2, y2 = map(int, box[:4])
                object_img = frame[y1:y2, x1:x2]
                object_class = names[int(class_id)]
                os.makedirs(os.path.join(detected_objects_dir, object_class), exist_ok=True)
                object_img_path = os.path.join(detected_objects_dir, object_class, f'frame{frame_count}_obj{i}.jpg')
                cv2.imwrite(object_img_path, object_img)

        results.render()

        out.write(frame)

    cap.release()
    out.release()

    st.write("Processing complete!")
    st.write(f"Detected objects saved in: {detected_objects_dir}")
    
    
    obj = os.listdir(detected_objects_dir)
    answer=[]
    for objects in obj:
        obj_pth = os.path.join(detected_objects_dir, objects)
        frame = sorted(os.listdir(obj_pth))
        frame_pth = os.path.join(obj_pth, frame[0])
        similar = main(frame_pth,yolo_model)
        for ss in similar:
            answer.append(ss)
    
    
    st.title("Product Details")

    products = answer
    # Loop through each product
    for product in products:
        name, image_url, product_link, rate, price = product

        # Create columns for layout
        col1, col2 = st.columns([1, 2])

        # Display the image in the first column
        with col1:
            st.image(image_url, use_column_width=True)
    
        # Display the details in the second column
        with col2:
            st.header(name)
            st.write(f"**Price:** {price}")
            st.write(f"**Rating:** {rate} / 5.0")
        
            # Create a button that links to the product page
            if st.button(f"Go to Amazon", key=name):
                st.markdown(f"[Click here to view the product]({product_link})")
        
        # Add a separator between products
        st.markdown("---")
        
    
