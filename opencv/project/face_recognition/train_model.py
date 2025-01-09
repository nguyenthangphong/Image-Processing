import cv2
import dlib
import numpy as np

# Load models
landmarks_model_path = "/home/phong/work/Image-Processing/opencv/project/train_face_recognition_model/models/shape_predictor_68_face_landmarks.dat"
face_recognition_model_path = "/home/phong/work/Image-Processing/opencv/project/train_face_recognition_model/models/dlib_face_recognition_resnet_model_v1.dat"

# Initialize face detector, shape predictor, and face recognition model
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(landmarks_model_path)
face_recognition_model = dlib.face_recognition_model_v1(face_recognition_model_path)

def detect_faces(image_path):
    try:
        # Read image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Unable to load image!")
            return False
        
        # Convert the image from OpenCV to dlib format (BGR to RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        faces = detector(rgb_image)

        if len(faces) == 0:
            print("No faces detected.")
            return False
        
        for face in faces:
            # Find the landmarks for the detected face
            shape = sp(rgb_image, face)

            # Get the face chip (aligned face region)
            face_chip = dlib.get_face_chip(rgb_image, shape)

            # Compute the face descriptor
            face_descriptor = face_recognition_model.compute_face_descriptor(face_chip)
            print(f"Face descriptor: {face_descriptor}")
        
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


# Example usage
image_path = "/home/phong/work/Image-Processing/opencv/images/face_1.jpg"
detect_faces(image_path)
