from ultralytics import YOLO
import time
import streamlit as st
import cv2
import settings
import threading

def sleep_and_clear_success():
    time.sleep(3)
    st.session_state['recyclable_placeholder'].empty()
    st.session_state['non_recyclable_placeholder'].empty()
    st.session_state['hazardous_placeholder'].empty()

def load_model(model_path):
    model = YOLO(model_path)
    return model

def classify_waste_type(detected_items):
    recyclable_items = set(detected_items) & set(settings.RECYCLABLE)
    non_recyclable_items = set(detected_items) & set(settings.NON_RECYCLABLE)
    hazardous_items = set(detected_items) & set(settings.HAZARDOUS)
    
    return recyclable_items, non_recyclable_items, hazardous_items

def _display_detected_frames(model, st_frame, image):
    image = cv2.resize(image, (720, int(720*(9/16))))
    
    if 'unique_classes' not in st.session_state:
        st.session_state['unique_classes'] = set()

    if 'recyclable_placeholder' not in st.session_state:
        st.session_state['recyclable_placeholder'] = st.sidebar.empty()
    if 'non_recyclable_placeholder' not in st.session_state:
        st.session_state['non_recyclable_placeholder'] = st.sidebar.empty()
    if 'hazardous_placeholder' not in st.session_state:
        st.session_state['hazardous_placeholder'] = st.sidebar.empty()

    if 'last_detection_time' not in st.session_state:
        st.session_state['last_detection_time'] = 0

    res = model.track(image, conf=0.5, persist=True, tracker='botsort.yaml')
    names = model.names
    detected_items = set()

    for result in res:
        new_classes = set([names[int(c)] for c in result.boxes.cls])
        
        if new_classes != st.session_state['unique_classes']:
            st.session_state['unique_classes'] = new_classes
            st.session_state['recyclable_placeholder'].info("Recyclable items: not detected") 
            st.session_state['non_recyclable_placeholder'].warning("Non-Recyclable items: not detected")
            st.session_state['hazardous_placeholder'].error("Hazardous items: not detected")
            detected_items.update(st.session_state['unique_classes'])

            recyclable_items, non_recyclable_items, hazardous_items = classify_waste_type(detected_items)

            if recyclable_items:
                detected_items_str = "\n- ".join(recyclable_items)
                st.session_state['recyclable_placeholder'].info(f"Recyclable items:\n- {detected_items_str}")
            if non_recyclable_items:
                detected_items_str = "\n- ".join(non_recyclable_items)
                st.session_state['non_recyclable_placeholder'].warning(f"Non-Recyclable items:\n- {detected_items_str}")
            if hazardous_items:
                detected_items_str = "\n- ".join(hazardous_items)
                st.session_state['hazardous_placeholder'].error(f"Hazardous items:\n- {detected_items_str}")

            threading.Thread(target=sleep_and_clear_success).start()
            st.session_state['last_detection_time'] = time.time()

    res_plotted = res[0].plot()
    st_frame.image(res_plotted, channels="BGR")


def play_webcam(model):
    source_webcam = settings.WEBCAM_PATH
    if st.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(model,st_frame,image)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
