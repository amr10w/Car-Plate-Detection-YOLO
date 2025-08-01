import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Cache the model loading to improve performance
@st.cache_resource
def load_model(model_path: str) -> YOLO:
    
    model = YOLO(model_path)
    return model

def process_image(image: Image.Image, model: YOLO, confidence_threshold: float) -> np.ndarray:
    
    # Run YOLO model inference with the specified confidence
    results = model(image, conf=confidence_threshold)
    
    # Plot the results on the image
    annotated_image = results[0].plot()
    
    return annotated_image

def process_video(video_path: str, model: YOLO, confidence_threshold: float, output_path: str = "output.mp4") -> str:
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return None
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the video codec and create VideoWriter object. 'avc1' is H.264, more compatible for web.
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add a progress bar for user feedback
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO model inference on the current frame
        results = model(frame, conf=confidence_threshold)
        annotated_frame = results[0].plot()
        
        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Update progress
        frame_count += 1
        if total_frames > 0:
            progress_bar.progress(frame_count / total_frames)
    
    
    cap.release()
    out.release()
    progress_bar.empty()  
    return output_path

def main():
   
    st.set_page_config(
        page_title="License Plate Recognition",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚗 License Plate Recognition with YOLO")
    st.markdown("""
    Welcome! Upload an image or video to detect license plates. 
    You can adjust the model path and confidence threshold in the sidebar.
    """)
    
    with st.sidebar:
        st.header("⚙️ Settings")
        model_path = st.text_input("Model Path", value="best.pt")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    uploaded_file = st.file_uploader(
        "Choose an image or video...",
        type=["jpg", "jpeg", "png", "mp4"],
        help="Supported formats: JPG, JPEG, PNG for images; MP4 for videos"
    )
    
    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        try:
            model = load_model(model_path)
            
            if file_extension in [".jpg", ".jpeg", ".png"]:
                # --- IMAGE PROCESSING LOGIC ---
                image = Image.open(uploaded_file)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("Detect License Plate in Image", key="image_button"):
                    with st.spinner("Processing image..."):
                        annotated_image = process_image(image, model, confidence_threshold)
                        with col2:
                            st.image(annotated_image, caption="Processed Image", use_column_width=True)
            
            elif file_extension == ".mp4":
                # --- VIDEO PROCESSING LOGIC ---
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                    tfile.write(uploaded_file.read())
                    temp_video_path = tfile.name
                
                st.video(temp_video_path, format="video/mp4")
                
                if st.button("Detect License Plate in Video", key="video_button"):
                    with st.spinner("Processing video... This may take a while."):
                        output_video_path = process_video(temp_video_path, model, confidence_threshold)
                        
                        st.success("✅ Video processing complete!")
                        
                        # Read the video file to a buffer after checking it exists and is not empty
                        if output_video_path and os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                            with open(output_video_path, 'rb') as video_file:
                                video_bytes = video_file.read()
                            st.video(video_bytes, format="video/mp4")
                            
                            # Clean up the temporary output file
                            os.unlink(output_video_path)
                        else:
                            st.error("❌ Processed video file could not be created or found. The video codec might not be supported on this system.")
                    
                    # Clean up the temporary input file
                    if os.path.exists(temp_video_path):
                        os.unlink(temp_video_path)
        
        except FileNotFoundError:
            st.error(f"Model file not found at '{model_path}'. Please check the path in the sidebar.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please ensure the model file is compatible and try again.")

if __name__ == "__main__":
    main()
