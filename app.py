import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import os
import sys

# Add scripts directory to path to import YOLODetector
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from detect import YOLODetector, load_config

# Set page config
st.set_page_config(
    page_title="Chili Disease Detector",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .stExpander {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_detector():
    config = load_config()
    model_cfg = config["model"]
    detect_cfg = config["detection"]
    
    color_map = {
        int(k): tuple(v)
        for k, v in detect_cfg.get("color_map", {}).items()
    }
    
    detector = YOLODetector(
        model_path=model_cfg["weights_path"],
        confidence_threshold=model_cfg["confidence_threshold"],
        color_map=color_map,
        quadrant_config=detect_cfg.get("quadrant", {}),
    )
    return detector, config

def main():
    st.title("🌶️ Chili Disease Detector")
    st.markdown("---")

    detector, config = get_detector()

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        conf_threshold = st.slider(
            "Confidence Threshold", 0.0, 1.0, 
            float(config["model"]["confidence_threshold"])
        )
        detector.confidence_threshold = conf_threshold
        
        st.markdown("---")
        st.write("### ESP32 Control")
        enable_esp32 = st.toggle("Enable ESP32 Serial Commands", value=False)
        if enable_esp32:
            port = st.text_input("Serial Port", value=config["serial"]["port"])
            if st.button("Test ESP32 Connection"):
                st.info(f"Connecting to {port}...")
                # We won't actually send a spray command here, just a test or log
                st.success("Configured for Serial communication.")

    # Main UI layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose a chili plant image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Load and display original image
            image_bytes = uploaded_file.read()
            detector.load_image_from_buffer(image_bytes)
            
            st.image(cv.cvtColor(detector.image, cv.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
            
            if st.button("Run Detection"):
                with st.spinner("Processing..."):
                    detector.detect_objects()
                    st.session_state["detected"] = True
                    st.session_state["area"] = detector.determine_area()

    with col2:
        st.subheader("Detection Results")
        if "detected" in st.session_state and st.session_state["detected"]:
            # Display result image
            res_img = cv.cvtColor(detector.image_all_boxes, cv.COLOR_BGR2RGB)
            st.image(res_img, caption="Detected Diseases", use_container_width=True)
            
            # Show stats
            st.success(f"Detections found: {len(detector.centers)}")
            
            if st.session_state["area"]:
                st.info(f"Recommended Spray Quadrant: **Area {st.session_state['area'][0]}**")
                
                if enable_esp32:
                    if st.button("Send Spray Command"):
                        success = detector.send_to_esp32(
                            st.session_state["area"],
                            port=port,
                            baud_rate=config["serial"]["baud_rate"]
                        )
                        if success:
                            st.success(f"Command sent to ESP32 on {port}")
                        else:
                            st.error("Failed to communicate with ESP32. Check connection and port.")
            else:
                st.warning("No disease detected. No spray action required.")
        else:
            st.info("Upload an image and click 'Run Detection' to see results.")

if __name__ == "__main__":
    main()
