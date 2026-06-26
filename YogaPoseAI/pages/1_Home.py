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

st.title("🧘 Yoga Pose Detection & Correction")

st.write("""
Welcome to YogaPose AI.

This system:
- Detects yoga poses
- Calculates confidence
- Gives corrections
- Tracks performance
""")

col1,col2,col3 = st.columns(3)

with col1:
    st.metric("Supported Poses","10")

with col2:
    st.metric("AI Model","PyTorch")

with col3:
    st.metric("Detection","Real Time")