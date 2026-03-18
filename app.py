import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import os
import sys
import time

# Add scripts directory to path to import YOLODetector
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from detect import YOLODetector, load_config

# Set page config
st.set_page_config(
    page_title="ChiliGuard AI | Disease Detector",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stApp {
        background-attachment: fixed;
    }
    
    /* Custom Card Style */
    .st-emotion-cache-12w0qpk {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        padding: 2rem;
    }
    
    /* Header Styling */
    h1 {
        color: #1e293b;
        font-weight: 800 !important;
        letter-spacing: -0.025em;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background: linear-gradient(90deg, #ff4b2b 0%, #ff416c 100%);
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 65, 108, 0.4);
    }
    
    /* Sidebar Styling */
    .stSidebar {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ff4b2b !important;
        border-bottom-color: #ff4b2b !important;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #ff4b2b;
    }
    
    /* Animation for Alerts */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .disease-alert {
        animation: pulse 2s infinite;
        background-color: #fee2e2;
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 1rem;
        color: #991b1b;
        font-weight: 600;
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
    detector, config = get_detector()

    # Sidebar
    with st.sidebar:
        if os.path.exists("assets/logo.png"):
            st.image("assets/logo.png", use_container_width=True)
        else:
            st.title("🌶️ ChiliGuard AI")
            
        st.markdown("---")
        st.subheader("⚙️ Analysis Settings")
        conf_threshold = st.slider(
            "Confidence Threshold", 0.1, 1.0, 
            float(config["model"]["confidence_threshold"])
        )
        detector.confidence_threshold = conf_threshold
        
        st.markdown("---")
        st.subheader("🔌 Hardware Interface")
        enable_esp32 = st.toggle("Connect to ESP32 Serial", value=False)
        if enable_esp32:
            port = st.text_input("Serial Port", value=config["serial"]["port"])
            st.info(f"Targeting port: {port}")
        
        st.markdown("---")
        st.caption("Powered by YOLOv8 & Streamlit")

    # Main Content Area
    st.title("ChiliGuard AI Analysis Dashboard")
    st.markdown("Automated disease detection and localized pesticide control.")

    tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📊 Data Insights", "ℹ️ Documentation"])

    with tab1:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.subheader("📥 Image Source")
            uploaded_file = st.file_uploader("Upload a chili plant image (JPG, PNG)", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image_bytes = uploaded_file.read()
                detector.load_image_from_buffer(image_bytes)
                
                # Use a container for the image with a subtle border
                st.image(cv.cvtColor(detector.image, cv.COLOR_BGR2RGB), 
                         caption="Original Input", 
                         use_container_width=True)
                
                if st.button("✨ START ANALYSIS"):
                    with st.spinner("🧠 AI is analyzing the foliage..."):
                        time.sleep(0.5) # Smooth UX
                        detector.detect_objects()
                        st.session_state["detected"] = True
                        st.session_state["area"] = detector.determine_area()
                        st.session_state["num_detections"] = len(detector.centers)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.subheader("🎯 Result Visualization")
            if "detected" in st.session_state and st.session_state["detected"]:
                # Display result image
                res_img = cv.cvtColor(detector.image_all_boxes, cv.COLOR_BGR2RGB)
                st.image(res_img, caption="AI Detection Layer", use_container_width=True)
                
                # Metrics Section
                mc_col1, mc_col2 = st.columns(2)
                mc_col1.metric("Leaves Detected", st.session_state["num_detections"])
                
                if st.session_state["area"]:
                    area_num = st.session_state["area"][0]
                    mc_col2.metric("Target Area", f"Quadrant {area_num}")
                    
                    st.markdown(f"""
                    <div class="disease-alert">
                        ⚠️ DISEASE DETECTED: Localized infection identified in <b>Area {area_num}</b>.
                        Ready for targeted spraying.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if enable_esp32:
                        st.markdown("---")
                        if st.button("🚀 ACTIVATE SPRAY SYSTEM"):
                            with st.spinner("Sending command..."):
                                success = detector.send_to_esp32(
                                    st.session_state["area"],
                                    port=port,
                                    baud_rate=config["serial"]["baud_rate"]
                                )
                                if success:
                                    st.balloons()
                                    st.success(f"Command successfully broadcast to ESP32 on {port}")
                                else:
                                    st.error("❌ Communication Failure. Please verify hardware link.")
                else:
                    st.success("✅ HEALTHY FOLIAGE: No disease patterns recognized.")
            else:
                st.info("Performance stats and detection masks will appear here after analysis.")

    with tab2:
        st.subheader("System Performance & Logs")
        st.info("Historical data tracking and performance metrics are under development.")
        
        # Example metrics for demo purposes
        p_col1, p_col2, p_col3 = st.columns(3)
        p_col1.metric("Detection Latency", "124ms")
        p_col2.metric("Model Recall", "92.4%")
        p_col3.metric("Pesticide Saved", "78%")

    with tab3:
        st.subheader("Knowledge Center")
        st.markdown("""
        ### How it works
        1. **Feature Extraction**: The YOLOv8 model extracts hierarchical features from the image.
        2. **Spatial Mapping**: Detections are mapped to a 2x2 grid based on your `config.yaml` split points.
        3. **Localized Action**: Only the specific solenoid corresponding to the infected area is activated.
        
        ### Troubleshooting
        - **Camera Drift**: Verify `x_split` and `y_split` in configuration.
        - **Serial Failure**: Ensure your user account has permissions for the COM port.
        """)

if __name__ == "__main__":
    main()
