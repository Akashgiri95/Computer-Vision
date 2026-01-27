import cv2
import numpy as np

def initialize_cascades():
    """Initialize all Haar Cascade classifiers"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )
    mouth_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_smile.xml'
    )
    return face_cascade, eye_cascade, mouth_cascade

def process_frame(
    frame,
    face_cascade,
    eye_cascade,
    mouth_cascade,
    face_scale_factor=1.3,
    face_min_neighbors=5,
    eye_scale_factor=1.3,
    eye_min_neighbors=5,
    mouth_scale_factor=1.8,
    mouth_min_neighbors=20,
    detect_faces=True,
    detect_eyes=True,
    detect_mouths=True
):
    """
    Process a single frame with given detection parameters
    
    Parameters:
    - frame: Input image frame (BGR format)
    - All cascade classifiers
    - Detection parameters for face, eye, mouth
    - Flags to enable/disable detections
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create a copy of the original frame for drawing
    result_frame = frame.copy()
    
    # Initialize detection counters
    detection_counts = {
        'faces': 0,
        'eyes': 0,
        'mouths': 0
    }
    
    # Face Detection
    if detect_faces:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=face_scale_factor,
            minNeighbors=face_min_neighbors
        )
        detection_counts['faces'] = len(faces)
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result_frame, "Face", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Extract face ROI
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = result_frame[y:y+h, x:x+w]
            
            # Eye Detection (within face ROI)
            if detect_eyes:
                eyes = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=eye_scale_factor,
                    minNeighbors=eye_min_neighbors
                )
                detection_counts['eyes'] += len(eyes)
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                    cv2.putText(roi_color, "Eye", (ex, ey - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Mouth Detection (within face ROI)
            if detect_mouths:
                mouths = mouth_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=mouth_scale_factor,
                    minNeighbors=mouth_min_neighbors
                )
                detection_counts['mouths'] += len(mouths)
                
                for (mx, my, mw, mh) in mouths:
                    cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
                    cv2.putText(roi_color, "Mouth", (mx, my - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Convert BGR to RGB for display
    result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
    
    return result_frame_rgb, detection_counts

def get_frame_from_webcam():
    """Capture a single frame from webcam"""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return None
    
    ret, frame = cam.read()
    cam.release()
    
    if ret:
        return frame
    return None