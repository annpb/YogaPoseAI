import streamlit as st
st.image(
    "assets/logo.png",
    width=150
)
with open("assets/styles.css") as f:

    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

st.title("ℹ️ About")

st.write("""
Yoga Pose Detection and Correction System

Technology Used:
- Python
- MediaPipe
- PyTorch
- OpenCV
- Streamlit
""")