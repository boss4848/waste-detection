from pathlib import Path
import streamlit as st
import helper
import settings

st.set_page_config(
    page_title="Waste Detection",
)

st.sidebar.title("Detect Console")

model_path = Path(settings.DETECTION_MODEL)

st.title("Intelligent waste segregation system")
st.write("Start detecting objects in the webcam stream by clicking the button below. To stop the detection, click stop button in the top right corner of the webcam stream.")

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
helper.play_webcam(model)

st.sidebar.markdown("This is a demo of the waste detection model.", unsafe_allow_html=True)

