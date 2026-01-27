import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from face_detection_backend import initialize_cascades, process_frame, get_frame_from_webcam

# Page configuration
st.set_page_config(
    page_title="Face Detection Parameter Tuning",
    page_icon="😊",
    layout="wide"
)

# Title and description
st.title("🎭 Face Detection Parameter Tuner")
st.markdown("""
Adjust the parameters below to see how they affect face, eye, and mouth detection.
The real-time webcam feed will show the detection results with different colors:
- **Faces**: Green rectangles
- **Eyes**: Blue rectangles  
- **Mouths**: Red rectangles
""")

# Initialize session state for webcam
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False
if 'last_frame' not in st.session_state:
    st.session_state.last_frame = None

# Sidebar for parameters
st.sidebar.header("🔧 Detection Parameters")

# Toggle switches for detection types
st.sidebar.subheader("Detection Toggles")
detect_faces = st.sidebar.checkbox("Detect Faces", value=True)
detect_eyes = st.sidebar.checkbox("Detect Eyes", value=True)
detect_mouths = st.sidebar.checkbox("Detect Mouths", value=True)

# Face Detection Parameters
st.sidebar.subheader("Face Detection")
face_scale_factor = st.sidebar.slider(
    "Face Scale Factor",
    min_value=1.01,
    max_value=2.0,
    value=1.3,
    step=0.01,
    help="Lower values detect more faces but slower, higher values faster but may miss faces"
)

face_min_neighbors = st.sidebar.slider(
    "Face Min Neighbors",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
    help="Higher values reduce false positives but may miss faces"
)

# Eye Detection Parameters
st.sidebar.subheader("Eye Detection")
eye_scale_factor = st.sidebar.slider(
    "Eye Scale Factor",
    min_value=1.01,
    max_value=2.0,
    value=1.3,
    step=0.01,
    help="Adjust for eye detection sensitivity"
)

eye_min_neighbors = st.sidebar.slider(
    "Eye Min Neighbors", 
    min_value=1,
    max_value=10,
    value=5,
    step=1,
    help="Controls false eye detections"
)

# Mouth Detection Parameters
st.sidebar.subheader("Mouth Detection")
mouth_scale_factor = st.sidebar.slider(
    "Mouth Scale Factor",
    min_value=1.01,
    max_value=2.0,
    value=1.8,
    step=0.01,
    help="Mouths vary in size, usually needs higher scale factor"
)

mouth_min_neighbors = st.sidebar.slider(
    "Mouth Min Neighbors",
    min_value=5,
    max_value=40,
    value=20,
    step=1,
    help="Higher values reduce false mouth detections"
)

# Information panel
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Parameter Information")

st.sidebar.markdown("""
### **ScaleFactor (1.01-2.0)**
- **Low (1.05)**: More detections, slower
- **Medium (1.3)**: Balanced (recommended)
- **High (1.8)**: Faster, may miss detections

### **MinNeighbors (1-40)**
- **Low (1-3)**: Many detections, many false positives
- **Medium (5-10)**: Balanced
- **High (15+)**: Strict, reduces false positives
""")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Webcam control
    st.subheader("📷 Webcam Feed")
    
    # Webcam control buttons
    col_buttons = st.columns([1, 1, 2])
    with col_buttons[0]:
        if st.button("🎥 Start Camera"):
            st.session_state.camera_on = True
    with col_buttons[1]:
        if st.button("⏸️ Stop Camera"):
            st.session_state.camera_on = False
    
    # Image placeholder
    image_placeholder = st.empty()
    
    # Detection stats placeholder
    stats_placeholder = st.empty()

with col2:
    st.subheader("📈 Parameter Effects")
    
    # Current parameter display
    st.markdown(f"""
    ### Current Settings:
    **Face Detection:**
    - Scale Factor: `{face_scale_factor}`
    - Min Neighbors: `{face_min_neighbors}`
    
    **Eye Detection:**
    - Scale Factor: `{eye_scale_factor}`
    - Min Neighbors: `{eye_min_neighbors}`
    
    **Mouth Detection:**
    - Scale Factor: `{mouth_scale_factor}`
    - Min Neighbors: `{mouth_min_neighbors}`
    """)
    
    # Tips section
    st.markdown("### 💡 Tips for Tuning")
    tips = """
    1. **If missing detections**: Decrease MinNeighbors or ScaleFactor
    2. **If too many false positives**: Increase MinNeighbors
    3. **For faster detection**: Increase ScaleFactor
    4. **Mouth detection**: Usually needs higher MinNeighbors (15-25)
    5. **Eye detection**: Works best with smaller ScaleFactor (1.1-1.3)
    """
    st.info(tips)

# Initialize cascades once
@st.cache_resource
def get_cascades():
    return initialize_cascades()

face_cascade, eye_cascade, mouth_cascade = get_cascades()

# Main processing loop
if st.session_state.camera_on:
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Camera status
    st.info("Camera is active. Adjust parameters in real-time!")
    
    # Initialize camera
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    fps_placeholder = st.empty()
    
    try:
        while st.session_state.camera_on:
            start_time = time.time()
            
            # Read frame
            ret, frame = cam.read()
            if not ret:
                st.error("Failed to grab frame from camera")
                break
            
            # Process frame with current parameters
            processed_frame, detection_counts = process_frame(
                frame,
                face_cascade,
                eye_cascade,
                mouth_cascade,
                face_scale_factor,
                face_min_neighbors,
                eye_scale_factor,
                eye_min_neighbors,
                mouth_scale_factor,
                mouth_min_neighbors,
                detect_faces,
                detect_eyes,
                detect_mouths
            )
            
            # Convert to PIL Image for display
            pil_image = Image.fromarray(processed_frame)
            
            # Update the image
            image_placeholder.image(pil_image, caption="Live Detection", use_column_width=True)
            
            # Update stats
            stats_text = f"""
            ### 🔍 Detection Counts:
            - **Faces**: {detection_counts['faces']}
            - **Eyes**: {detection_counts['eyes']}  
            - **Mouths**: {detection_counts['mouths']}
            """
            stats_placeholder.markdown(stats_text)
            
            # Calculate and display FPS
            frame_count += 1
            fps = 1.0 / (time.time() - start_time)
            fps_placeholder.metric("FPS", f"{fps:.1f}")
            
            # Update progress bar (cycling)
            progress_bar.progress((frame_count % 100) / 100)
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
            
    except Exception as e:
        st.error(f"Error: {e}")
    
    finally:
        # Release camera
        cam.release()
        st.session_state.camera_on = False
        st.rerun()

else:
    # Show static image when camera is off
    if st.button("📸 Capture Single Frame"):
        frame = get_frame_from_webcam()
        if frame is not None:
            # Process the single frame
            processed_frame, detection_counts = process_frame(
                frame,
                face_cascade,
                eye_cascade,
                mouth_cascade,
                face_scale_factor,
                face_min_neighbors,
                eye_scale_factor,
                eye_min_neighbors,
                mouth_scale_factor,
                mouth_min_neighbors,
                detect_faces,
                detect_eyes,
                detect_mouths
            )
            
            # Convert to PIL Image
            pil_image = Image.fromarray(processed_frame)
            
            # Display
            image_placeholder.image(pil_image, caption="Single Frame Detection", use_column_width=True)
            
            # Show stats
            stats_text = f"""
            ### 🔍 Detection Counts:
            - **Faces**: {detection_counts['faces']}
            - **Eyes**: {detection_counts['eyes']}  
            - **Mouths**: {detection_counts['mouths']}
            """
            stats_placeholder.markdown(stats_text)
        else:
            st.error("Could not capture frame from camera")

# Footer
st.markdown("---")
st.markdown("""
### 🎯 How to Use:
1. Click **'Start Camera'** to begin live detection
2. Adjust sliders to see parameter effects in real-time
3. Use checkboxes to toggle detection types
4. Click **'Stop Camera'** to pause
5. Use **'Capture Single Frame'** for static testing
""")