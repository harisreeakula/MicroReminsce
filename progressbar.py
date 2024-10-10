import streamlit as st
import cv2
import os
import numpy as np
import tempfile
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics.pairwise import cosine_similarity
import face_recognition
import logging
import glob

# Initialize the ResNet50 model
@st.cache_resource
def load_model():
    return ResNet50(weights='imagenet', include_top=False, pooling='avg')

model = load_model()

# App 1: Object Detection in Video
def extract_features(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

def capture_frames(video_path, output_path, frame_interval=1, progress_bar=None):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    saved_count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_name = os.path.join(output_path, f"frame_{saved_count:06d}.jpg")
            small_frame = cv2.resize(frame, (224, 224))
            cv2.imwrite(frame_name, small_frame)
            saved_count += 1
        
        frame_count += 1

        # Update progress bar
        if progress_bar:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
        
    video.release()
    return fps

def detect_object_in_frames(frames_path, reference_image_path, similarity_threshold=0.8):
    reference_features = extract_features(reference_image_path)
    frames = sorted(glob.glob(os.path.join(frames_path, '*.jpg')))
    detected_frames = []
    
    for filename in frames:
        frame_features = extract_features(filename)
        similarity = cosine_similarity(reference_features.reshape(1, -1), frame_features.reshape(1, -1))[0][0]
        
        if similarity >= similarity_threshold:
            frame_number = int(os.path.splitext(os.path.basename(filename))[0].split('_')[1])
            detected_frames.append(frame_number)
    
    return detected_frames

def get_object_durations(detected_frames, fps, frame_interval):
    if not detected_frames:
        return []
    
    durations = []
    start = detected_frames[0]
    prev = start
    
    for frame in detected_frames[1:] + [None]:
        if frame is None or frame - prev > 1:
            end = prev
            durations.append((start * frame_interval / fps, (end + 1) * frame_interval / fps))
            start = frame
        prev = frame
    
    return durations

def object_detection_app():
    st.title("Object track")

    video_file = st.file_uploader("Upload the video file", type=["mp4", "avi", "mov"])
    reference_image = st.file_uploader("Upload the reference image", type=["jpg", "jpeg", "png"])

    frame_interval = st.slider("Frame interval (process every nth frame)", min_value=1, max_value=30, value=5)

    if st.button("Run Object Detection"):
        if video_file and reference_image:
            with tempfile.TemporaryDirectory() as temp_dir:
                video_temp_path = os.path.join(temp_dir, video_file.name)
                ref_image_temp_path = os.path.join(temp_dir, reference_image.name)

                with open(video_temp_path, "wb") as f:
                    f.write(video_file.read())
                with open(ref_image_temp_path, "wb") as f:
                    f.write(reference_image.read())

                # Initialize progress bar
                st.write("Processing video...")
                progress_bar = st.progress(0)

                fps = capture_frames(video_temp_path, temp_dir, frame_interval, progress_bar)
                detected_frames = detect_object_in_frames(temp_dir, ref_image_temp_path)
                object_durations = get_object_durations(detected_frames, fps, frame_interval)

                if object_durations:
                    st.success("The object was detected during the following time intervals:")
                    for start, end in object_durations:
                        st.write(f"From {start:.2f} seconds to {end:.2f} seconds")
                else:
                    st.error("The object was not detected in the video.")
        else:
            st.warning("Please upload both a video and a reference image.")

# App 2: Criminal Face Identification in Video
def load_image(image_path):
    try:
        image = face_recognition.load_image_file(image_path)
        return image
    except Exception as e:
        logging.error(f"Error loading image: {e}")
        return None

def get_face_encoding(image):
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) == 0:
        return None
    return face_encodings[0]

def facial_recognition_with_tracking(criminal_image, video_path, progress_bar=None):
    criminal_face_encoding = get_face_encoding(criminal_image)
    if criminal_face_encoding is None:
        return None

    known_face_encodings = [criminal_face_encoding]
    known_face_names = ["Criminal"]

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        return None

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    criminal_appearances = []
    current_appearance_start = None

    criminal_frames = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        current_time = frame_count / fps

        if frame_count % 30 != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            if current_appearance_start is not None:
                criminal_appearances.append((current_appearance_start, current_time))
                current_appearance_start = None
            continue

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        criminal_detected = False
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                criminal_detected = True
                if current_appearance_start is None:
                    current_appearance_start = current_time
                criminal_frames.append(frame)
                break

        if not criminal_detected and current_appearance_start is not None:
            criminal_appearances.append((current_appearance_start, current_time))
            current_appearance_start = None

        # Update progress bar
        if progress_bar:
            progress_bar.progress(min(frame_count / total_frames, 1.0))

    video_capture.release()

    if current_appearance_start is not None:
        criminal_appearances.append((current_appearance_start, total_frames / fps))

    return criminal_appearances, total_frames / fps, criminal_frames

def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def criminal_face_identification_app():
    st.title("Face Track")

    criminal_image_file = st.file_uploader("Upload Person's Image", type=["jpg", "jpeg", "png"])
    video_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])

    if criminal_image_file is not None and video_file is not None:
        temp_image_path = os.path.join("temp", criminal_image_file.name)
        temp_video_path = os.path.join("temp", video_file.name)

        with open(temp_image_path, "wb") as f:
            f.write(criminal_image_file.read())
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())

        if st.button("Run"):
            criminal_image = load_image(temp_image_path)
            
            # Initialize progress bar
            st.write("Processing video...")
            progress_bar = st.progress(0)

            result = facial_recognition_with_tracking(criminal_image, temp_video_path, progress_bar)
            if result is not None:
                appearances, total_duration, criminal_frames = result
                if appearances:
                    st.write("Person appearances:")
                    for start, end in appearances:
                        start_time = format_time(start)
                        end_time = format_time(end)
                        duration = format_time(end - start)
                        st.write(f"From {start_time} to {end_time} (Duration: {duration})")

                    for frame in criminal_frames:
                        st.image(frame, channels="BGR")
                else:
                    st.write("Person not detected in the video.")
                st.write(f"Total video duration: {format_time(total_duration)}")
            else:
                st.error("An error occurred while processing the video.")

# Main Application
def main():
    st.sidebar.title("Micro Reminsce")
    app_options = ["Object Track", "Face Track"]
    app_choice = st.sidebar.selectbox("Choose an Application", app_options)

    if app_choice == "Object Track":
        object_detection_app()
    elif app_choice == "Face Track":
        criminal_face_identification_app()

if __name__ == '__main__':
    main()
