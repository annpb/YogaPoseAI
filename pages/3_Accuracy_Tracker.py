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
import pandas as pd


st.title("📈 Accuracy Tracker")

data = pd.DataFrame({
    "Session":[1,2,3,4,5],
    "Accuracy":[80,85,90,92,95]
})

st.line_chart(
    data.set_index("Session")
)