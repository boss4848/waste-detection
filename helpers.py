from ultralytics import YOLO
import time
import streamlit as st
import cv2
import settings

def display_tracker_options():
    # display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True
    if is_display_tracker:
        # tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, 'botsort.yaml'
    return is_display_tracker, None


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def _display_detected_frames(model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))
    # Display object tracking, if specified
    # Initialize session_state if it doesn't exist
    if 'unique_classes' not in st.session_state:
        st.session_state['unique_classes'] = set()

    # Create a placeholder for the unique classes
    if 'unique_classes_placeholder' not in st.session_state:
        st.session_state['unique_classes_placeholder'] = st.empty()

    if is_display_tracking:
        res = model.track(image, conf=0.5, persist=True, tracker=tracker)
        names = model.names
        # Display the class names on the web only once
        # st.write([names[int(c)] for c in res.boxes.cls])  
        for result in res:
            # Update the unique_classes set
            new_classes = set([names[int(c)] for c in result.boxes.cls])
            if new_classes != st.session_state['unique_classes']:
                st.session_state['unique_classes'] = new_classes
                # Update the placeholder's value
                st.session_state['unique_classes_placeholder'].text(st.session_state['unique_classes'])
                if st.session_state['unique_classes'] == {'person'}:
                    st.sidebar.info(
                    '''
                    ### Person Detected
                    - ไอเสื่อม
                    '''
                    )
        
                # if st.session_state['unique_classes'] == {'person'}:
                #     st.write('person detected')
                #     st.markdown("""
                #     <style>
                #         [data-testid=stSidebar] {
                #             background-color: #ff000050;
                #         }
                #     </style>
                #     """, unsafe_allow_html=True)

                #     with st.sidebar:
                #         "## This is the sidebar"
        # Clear the previous output and print the unique classes
        # place_holder = st.empty()

        # for result in res:
        #     for c in result.boxes.cls:
        #         print(names[int(c)])
        #         # Display the class names on the web 
        #         place_holder.text(names[int(c)])
        #         # st.write(names[int(c)])
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

def play_webcam(model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
