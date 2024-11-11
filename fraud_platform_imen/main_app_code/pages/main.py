#main.py

import sys
import streamlit as st
import os
import base64
import cv2
import numpy as np
from camera_input_live import camera_input_live
# Add the main_app_code directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ner_app

from PIL import Image
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
from tqdm import tqdm

import pandas as pd
import random
from spacy import displacy

# if "user_logged_in" not in st.session_state:
#     st.session_state.user_logged_in = False

# if not st.session_state.user_logged_in:
#     st.warning("Please log in to access this page.")
#     st.stop()  # Stop execution if the user is not logged in



def set_background_color(bg_color, font_color):
    return f"""
    <style>
    .background-text {{
        background-color: {bg_color};
        color: {font_color};
        padding: 10px;
        border-radius: 5px;
        font-family: Arial, sans-serif;
    }}
    </style>
    """

# Load custom spaCy NLP pipeline
nlp = spacy.load("my_saved_model")

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f"""
    <style>
    .stApp {{
    background-image: url("data:image/png;base64,{bin_str}");
    background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

def cooling_highlight(val):
    color = '#ACE5EE' if val else '#F2F7FA'
    return f'background-color: {color}'

st.title("Fake Medicine Detection App")
set_background(r'fraud_platform_imen\widgets\medicine-capsules.png')
st.write("Welcome to Fake Medicine Detection App")

# Sidebar configuration
with st.sidebar:
    st.image(r"fraud_platform_imen\widgets\medical-symbol.png", width=110)
    st.title("MediCare")
    choice = st.radio("Navigation", ["Extract text from images", "Display labeled text", "Fraud Detection"], index=0)
    st.info("This project application helps you annotate your medicine data and detect fraud.")
    st.sidebar.success("Select an option above.")

if choice == "Extract text from images":
    sub_choice = st.radio('Choose an option', ["Upload Image", "Capture Image with Camera"])

    if sub_choice == "Capture Image with Camera":
        st.write("# See a new image every second")
        controls = st.checkbox("Show controls")  # Show controls button to pause the video
        image = camera_input_live(show_controls=controls)

        if image is not None:
            st.image(image, caption="Captured Image", use_container_width=True)

            if st.button("Extract Text From Image"):
                numpy_array = np.array(image)
                opencv_image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)

                text, result = ner_app.ocr_extraction(opencv_image)
                st.session_state["extracted_text"] = text  # Save extracted text in session

                st.markdown("Here's the Extracted text:")
                st.markdown(set_background_color("#f2f7fa", 'black'), unsafe_allow_html=True)
                styled_text = f"<div class='background-text'>{text}</div>"
                st.markdown(styled_text, unsafe_allow_html=True)

                img_with_contours = ner_app.draw_contours(opencv_image, result)
                st.markdown("Here's the image with contours on the text detected:")
                st.image(img_with_contours, caption="Image with Contours", channels="BGR", use_container_width=True)

    elif sub_choice == "Upload Image":
        st.title("Upload Your Image")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            st.image(opencv_image, caption="Uploaded Image", channels="BGR", use_container_width=True)

            if st.button("Extract Text From Image"):
                text, result = ner_app.ocr_extraction(opencv_image)
                st.session_state["extracted_text"] = text  # Save extracted text in session

                st.markdown("Here's the Extracted text:")
                st.markdown(set_background_color("#f2f7fa", 'black'), unsafe_allow_html=True)
                styled_text = f"<div class='background-text'>{text}</div>"
                st.markdown(styled_text, unsafe_allow_html=True)

                img_with_contours = ner_app.draw_contours(opencv_image, result)
                st.markdown("Here's the image with contours on the text detected:")
                st.image(img_with_contours, caption="Image with Contours", channels="BGR", use_container_width=True)

elif choice == "Display labeled text":
    st.title('Named Entity Recognition')
    text = st.session_state.get("extracted_text", "")

    if not text.strip():
        st.warning("No text available for Named Entity Recognition")
        st.stop()

    if st.button("Perform Named Entity Recognition"):
        doc = ner_app.perform_named_entity_recognition(text)
        st.session_state["ner_doc"] = doc  # Save NER doc in session

        html = ner_app.display_doc(doc)
        if html is not None:
            styled_text = f"<div class='background-text'><h3>{html}</h3></div>"
            st.markdown(styled_text, unsafe_allow_html=True)

elif choice == "Fraud Detection":
    st.title("Fraud Detection with Jaccard Similarity")
    st.write('Detection of fraud in the text extracted:\n')

    if st.button("Calculation of Maximum Jaccard Score"):
        text = st.session_state.get("extracted_text", "")
        doc = ner_app.perform_named_entity_recognition(text)  # Process text as a Doc
        max_jaccard_score, entities, fraud_status = ner_app.fraud(doc)  # Pass Doc object

        # Display Max Jaccard Similarity Score
        st.markdown("Max Jaccard Similarity Score:")
        st.markdown(set_background_color("#f2f7fa", 'black'), unsafe_allow_html=True)
        styled_text = f"<div class='background-text'>{max_jaccard_score}</div>"
        st.markdown(styled_text, unsafe_allow_html=True)

        # Ensure entities is formatted as a DataFrame
        if isinstance(entities, pd.Series):
            # Convert Series to a DataFrame
            entities_df = pd.DataFrame([entities.to_dict()])
        elif isinstance(entities, dict):
            # Convert single dictionary to a DataFrame
            entities_df = pd.DataFrame([entities])
        elif isinstance(entities, list):
            if all(isinstance(i, pd.Series) for i in entities):
                # Convert list of Series to list of dictionaries
                entities_df = pd.DataFrame([i.to_dict() for i in entities])
            elif all(isinstance(i, dict) for i in entities):
                # Convert list of dictionaries directly to DataFrame
                entities_df = pd.DataFrame(entities)
            else:
                entities_df = None
        else:

            entities_df = None

        # Display entities if available
        if entities_df is not None:
            st.write(entities_df)

        # Display fraud status with appropriate color
        fraud_color = "#FF0000" if fraud_status == "This Drug is potentially fraudulent" else "#008000"
        styled_text = f"<div class='background-text'>{fraud_status}</div>"
        st.markdown(set_background_color(fraud_color, 'White'), unsafe_allow_html=True)
        st.markdown(styled_text, unsafe_allow_html=True)
