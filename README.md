# Micro Reminisce
Micro Reminisce is a Streamlit application that provides two functionalities:

Object Track - Detects and tracks a reference object in a video.

Face Track - Identifies and tracks a specific face (criminal or other) in a video.

Requirements
Ensure the following dependencies are installed:


pip install streamlit opencv-python-headless numpy tensorflow scikit-learn face-recognition
How to Run the Application
To run the Streamlit app, follow these steps:

Clone the repository or download the files.
Install the required libraries using the command above.
Run the Streamlit application using the following command:
bash
Copy code
streamlit run app.py
This will launch the application in your web browser.

Application Structure
The application contains two main modules:

1. Object Track
Upload a video file (.mp4, .avi, .mov) and a reference image (.jpg, .jpeg, .png).
The application processes the video and identifies frames where the reference object is detected based on cosine similarity of features extracted using the ResNet50 model.
Results are shown as time intervals during which the object is present in the video.


3. Face Track
Upload a person's image (.jpg, .jpeg, .png) and a video file (.mp4, .avi, .mov).
The app processes the video and identifies frames where the face is detected using face_recognition.
It provides time intervals when the person was present and shows frames of their appearance.


Application Flow


Object Detection App:
Upload the video and reference image.
Choose the frame interval for processing (how frequently frames are analyzed).
Click Run Object Detection to start processing.
The app will display time intervals during which the object appears in the video.


Criminal Face Identification App:
Upload the person's image and the video file.
Click Run to start the facial recognition process.
The app will display time intervals when the person appears and show frames where their face is detected.


Notes
This app is useful for tracking objects and individuals in video footage, particularly for security and surveillance purposes.
Progress bars are displayed while the video is being processed.
Customization
Modify similarity_threshold in the detect_object_in_frames function for fine-tuning object detection.
Adjust the frame interval for faster or more thorough processing.
License
This project is licensed under the MIT License.

