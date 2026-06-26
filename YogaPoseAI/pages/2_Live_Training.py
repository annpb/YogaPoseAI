import streamlit as st
import av
import cv2

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from backend.predictor import process_frame
import time

# -------------------------
# Session State
# -------------------------
if "pose_name" not in st.session_state:
    st.session_state.pose_name = "No Pose"

if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0

if "feedback" not in st.session_state:
    st.session_state.feedback = []

# -------------------------
# Logo
# -------------------------
st.image(
    "assets/logo.png",
    width=150
)

# -------------------------
# CSS
# -------------------------
with open("assets/styles.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# -------------------------
# Title
# -------------------------
st.title("🎥 Live Training")

# -------------------------
# Layout
# -------------------------
col1, col2 = st.columns([2, 1])

# -------------------------
# Webcam Processor
# -------------------------
class PoseProcessor(VideoProcessorBase):

    def __init__(self):
        self.pose_name = "No Pose"
        self.confidence = 0.0
        self.feedback = []

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        try:

            img, pose_name, confidence, feedback = process_frame(img)

            # Store results on the processor object (thread-safe local storage)
            self.pose_name = pose_name
            self.confidence = confidence
            self.feedback = feedback

        except Exception as e:

            self.pose_name = "Error"
            self.feedback = [str(e)]

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# -------------------------
# Webcam
# -------------------------
with col1:

    webrtc_ctx = webrtc_streamer(
        key="yoga",
        video_processor_factory=PoseProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

# -------------------------
# Metrics
# -------------------------
with col2:

    st.subheader("Live Pose Status")

    pose_placeholder = st.empty()
    conf_placeholder = st.empty()
    corr_placeholder = st.empty()

    # Continuously read values from the video processor and update UI
    if 'webrtc_ctx' in locals() and webrtc_ctx is not None:
        try:
            while True:
                vp = webrtc_ctx.video_processor
                if vp is None:
                    pose_placeholder.markdown("**Pose:** No Pose  \n**Confidence:** 0.00%")
                    corr_placeholder.info("No corrections yet. Move into the camera frame and hold a yoga pose.")
                else:
                    pose_placeholder.markdown(f"**Pose:** {vp.pose_name}  \n**Confidence:** {vp.confidence * 100:.2f}%")
                    if vp.feedback:
                        corr_placeholder.markdown("**Corrections:**")
                        for msg in vp.feedback:
                            corr_placeholder.error(msg)
                    else:
                        corr_placeholder.info("No corrections yet. Move into the camera frame and hold a yoga pose.")

                time.sleep(0.5)
        except Exception:
            # On page rerun or stop, exit the loop silently
            pass