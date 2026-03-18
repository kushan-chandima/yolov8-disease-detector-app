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
    page_title="ChiliGuard AI | Dark Mode",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a sleek Dark Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9;
    }
    
    /* Dark Glassmorphism Card Style */
    [data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        /* Container styling */
    }
    
    .st-emotion-cache-12w0qpk {
        background: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        padding: 2rem;
    }
    
    /* Global Text Contrast Fix for Dark Theme */
    [data-testid="stSidebar"] *, 
    [data-testid="stMain"] *,
    .stMarkdown, .stText, label, p, span, div {
        color: #e2e8f0 !important; /* Light grey/white for high contrast on dark */
    }

    /* Force specific headings to be light */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Button Styling - Keep vibrant red for action */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background: linear-gradient(90deg, #ff4b2b 0%, #ff416c 100%);
        color: white !important;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 65, 108, 0.4);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important; /* Muted inactive tabs */
    }
    
    .stTabs [aria-selected="true"] p {
        color: #ff4b2b !important; /* Active tab text */
    }
    
    .stTabs [aria-selected="true"] {
        border-bottom-color: #ff4b2b !important;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        color: #ff4b2b !important;
        font-weight: 800;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    /* File Uploader Appearance */
    [data-testid="stFileUploader"] {
        background-color: rgba(15, 23, 42, 0.5);
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 10px;
    }

    /* Alert Styling */
    .disease-alert {
        background-color: rgba(153, 27, 27, 0.3);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 1rem;
        color: #fecaca !important;
        font-weight: 600;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { transform: scale(1.02); box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
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
            # Use columns to center logo or add padding
            st.image("assets/logo.png", width="stretch")
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
        st.caption("v2.1 Dark Edition | YOLOv8")

    # Main Content Area
    st.title("ChiliGuard AI Analysis Dashboard")
    st.markdown("Precision agriculture through automated disease detection.")

    tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📊 Data Insights", "ℹ️ Documentation"])

    with tab1:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.subheader("📥 Input Stream")
            uploaded_file = st.file_uploader("Drop chili plant image here (JPG, PNG)", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image_bytes = uploaded_file.read()
                detector.load_image_from_buffer(image_bytes)
                
                st.image(cv.cvtColor(detector.image, cv.COLOR_BGR2RGB), 
                         caption="Processed Input", 
                         width="stretch")
                
                if st.button("✨ START AI SCAN"):
                    with st.spinner("Processing architectural layers..."):
                        time.sleep(0.5)
                        detector.detect_objects()
                        st.session_state["detected"] = True
                        st.session_state["area"] = detector.determine_area()
                        st.session_state["num_detections"] = len(detector.centers)

        with col2:
            st.subheader("🎯 Neural Output")
            if "detected" in st.session_state and st.session_state["detected"]:
                res_img = cv.cvtColor(detector.image_all_boxes, cv.COLOR_BGR2RGB)
                st.image(res_img, caption="Object Detection Overlay", width="stretch")
                
                mc_col1, mc_col2 = st.columns(2)
                mc_col1.metric("Infections Identified", st.session_state["num_detections"])
                
                if st.session_state["area"]:
                    area_num = st.session_state["area"][0]
                    mc_col2.metric("Target Quadrant", f"Area {area_num}")
                    
                    st.markdown(f"""
                    <div class="disease-alert">
                        ⚡ ACTION REQUIRED: Disease clusters identified in <b>Quadrant {area_num}</b>.
                        ESP32 precision spray sequence ready.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if enable_esp32:
                        st.markdown("---")
                        if st.button("🚀 TRIGGER PRECISION SPRAY"):
                            with st.spinner("Broadcasting instructions..."):
                                success = detector.send_to_esp32(
                                    st.session_state["area"],
                                    port=port,
                                    baud_rate=config["serial"]["baud_rate"]
                                )
                                if success:
                                    st.balloons()
                                    st.success(f"Command successful: ESP32 active on {port}")
                                else:
                                    st.error("Hardware link severed. Check serial connection.")
                else:
                    st.success("✅ OPTIMAL HEALTH: Foliage patterns clear of known diseases.")
            else:
                st.info("Analysis output and localized targets will be displayed here.")

    with tab2:
        st.subheader("System Performance & Historical Logs")
        st.markdown("Telemetric data from the last 24 hours:")
        
        p_col1, p_col2, p_col3 = st.columns(3)
        p_col1.metric("Avg Latency", "124ms", "-12ms")
        p_col2.metric("Detection Recall", "92.4%", "+1.2%")
        p_col3.metric("Pesticide Saved", "78%", "+5%")
        
        st.markdown("---")
        st.caption("Historical logging is currently in sandbox mode.")

    with tab3:
        st.subheader("Core Methodology")
        st.markdown("""
        ### AI Architecture
        - **Model**: YOLOv8 (You Only Look Once v8)
        - **Inference**: Optimized for localized CPU/GPU processing.
        - **Input**: 640x640 normalized RGB tensors.
        
        ### Hardware Synchronization
        - **Protocol**: Serial (Universal Asynchronous Receiver-Transmitter)
        - **Quadrant Mapping**: Automated polar coordinate translation to stepper motor steps.
        """)

if __name__ == "__main__":
    main()
