import streamlit as st
import pandas as pd

st.image(
    "assets/logo.png",
    width=150
)

with open("assets/styles.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

st.title("📚 Pose Library")

# -------------------------
# Load Poses from CSV
# -------------------------
try:
    label_df = pd.read_csv("models/label_map.csv")
    poses = label_df[['pose', 'label']].values.tolist()
    
    st.subheader(f"Supported Poses: {len(poses)}")
    
    # Display all poses
    cols = st.columns(2)
    
    for idx, (pose_id, pose_name) in enumerate(poses):
        with cols[idx % 2]:
            st.markdown(f"### {pose_name}")
            
            # Pose descriptions
            pose_info = {
                "Bakasana": "Crane Pose - Balance on hands with knees tucked",
                "Parsvottanasana": "Pyramid Pose - Forward fold with hands in prayer",
                "Pincha Mayurasana": "Forearm Stand - Advanced arm balance pose",
                "Trikonasana": "Triangle Pose - Standing pose with extended limbs",
                "Urdhva Dhanurasana": "Upward Bow Pose - Deep backbend on hands",
                "Ustrasana": "Camel Pose - Intense backbend on knees",
                "Utkatasana": "Chair Pose - Squatting position with arms up",
                "Uttanasana": "Forward Fold - Standing forward bend",
                "Utthita Parsvakonasana": "Extended Side Angle - Powerful standing pose",
                "Vrksasana": "Tree Pose - Balance pose standing on one leg"
            }
            
            description = pose_info.get(pose_name, "Traditional yoga pose")
            st.write(description)
            
except Exception as e:
    st.error(f"Error loading poses: {e}")
    st.info("Make sure models/label_map.csv exists")