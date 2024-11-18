import numpy as np
import os
import gdown  # Use gdown for reliable Google Drive downloads
import cv2
import spacy
from spacy.util import filter_spans
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
from spacy import displacy
import easyocr
import streamlit as st
import torch
from PIL import Image
import zipfile

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize EasyOCR and NLTK stopwords
reader = easyocr.Reader(['en'])
stop_words = set(stopwords.words("english"))

# Google Drive file IDs
CSV_FILE_ID = "1-eFFOJ4CSFAmIW607JIo2JVGt2aH4BE-" 
MODEL_FOLDER_ID = "1FjDkMYYbFlgzXLueAR2qCMVn7jtKJOie" 

# Local paths
CSV_PATH = "fraud_platform_imen/data/cleaned_medicine_data.csv"
MODEL_FOLDER_PATH = "my_saved_model" 

# Function to download file if needed
def download_file_if_needed(file_id, destination):
    """Download the file from Google Drive using gdown only if it doesn't already exist."""
    if not os.path.exists(destination):
        print(f"Downloading {destination} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", destination, quiet=False)
        print(f"{destination} downloaded successfully.")
    else:
        print(f"{destination} already exists locally. Skipping download.")

# Function to download the entire folder (in zip format)
def download_folder_if_needed(file_id, destination):
    """Download and extract the folder from Google Drive if it doesn't exist locally or is empty."""
    zip_file_path = f"{destination}.zip"
    
    # Check if the folder exists but is empty
    if not os.path.exists(destination) or not os.listdir(destination):
        print(f"Downloading {zip_file_path} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", zip_file_path, quiet=False)
        print(f"{zip_file_path} downloaded successfully.")
        
        # Check if the zip file is downloaded correctly
        if os.path.getsize(zip_file_path) == 0:
            print(f"Error: The downloaded ZIP file is empty. Please check the Google Drive file.")
            return
        
        # Extract the ZIP file
        print(f"Extracting {zip_file_path}...")
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # Get the names of the files/folders in the zip to ensure correct extraction
                zip_contents = zip_ref.namelist()
                
                # If the zip contains a top-level folder (e.g., 'my_saved_model'), extract it correctly
                if len(zip_contents) == 1 and zip_contents[0].startswith('my_saved_model/'):
                    # Extract without creating an extra folder
                    zip_ref.extractall(destination)
                else:
                    # Extract everything normally
                    zip_ref.extractall(destination)
                    
            print(f"{destination} extracted successfully.")
        except zipfile.BadZipFile as e:
            print(f"Error: Bad ZIP file: {e}")
        except Exception as e:
            print(f"Error extracting zip file: {e}")
        
        # Remove the zip file after extraction
        os.remove(zip_file_path)
        
        # Check the contents of the extracted folder
        check_extracted_files(destination)
    else:
        print(f"{destination} folder already exists and is not empty. Skipping download.")

# Function to check the contents of the extracted folder
def check_extracted_files(destination):
    """Print the contents of the folder after extraction."""
    if os.path.exists(destination):
        print(f"Contents of {destination}:")
        for root, dirs, files in os.walk(destination):
            for file in files:
                print(f"Found file: {file}")
    else:
        print(f"{destination} does not exist after extraction.")

# Ensure directories exist for CSV and model
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
os.makedirs(MODEL_FOLDER_PATH, exist_ok=True)

# Download CSV and model folder only if they are missing or empty
download_file_if_needed(CSV_FILE_ID, CSV_PATH)
download_folder_if_needed(MODEL_FOLDER_ID, MODEL_FOLDER_PATH)

# Load SpaCy model from the downloaded folder
try:
    nlp = spacy.load(MODEL_FOLDER_PATH)  # Load the SpaCy model from the local 'my_saved_model' folder
    print(f"Successfully loaded the SpaCy model from {MODEL_FOLDER_PATH}")
except Exception as e:
    print(f"Error loading SpaCy model: {e}")

#--------------------------------- Image Preprocessing -----------------------------------
def preprocess_image(opencv_image):
    """Preprocess image by converting to grayscale, applying Gaussian Blur, and adaptive thresholding."""
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    processed_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return processed_image

#--------------------------------- OCR Extraction -----------------------------------
def ocr_extraction(opencv_image):
    """Extracts text from an image using EasyOCR."""
    processed_image = preprocess_image(opencv_image)
    result = reader.readtext(processed_image, paragraph="False")
    
    text = ' '.join([res[1] for res in result])
    return text, result

def draw_contours(opencv_image, result):
    """Draws bounding boxes around detected text regions in the image."""
    for detection in result:
        top_left, bottom_right = tuple(detection[0][0]), tuple(detection[0][2])
        opencv_image = cv2.rectangle(opencv_image, top_left, bottom_right, (0, 255, 0), 3)
    return opencv_image

#--------------------------------- Named Entity Recognition -----------------------------------
def perform_named_entity_recognition(text):
    """Performs Named Entity Recognition (NER) on input text."""
    if isinstance(text, str):
        doc = nlp(text)
        return doc
    raise ValueError("Input to NER must be a string.")

def color_gen():
    """Generates a random hex color code."""
    return f'#{random.randint(0, 0xFFFFFF):06x}'

def display_doc(doc):
    """Renders NER entities in HTML format with random colors for Streamlit display."""
    colors = {ent.label_: color_gen() for ent in doc.ents}
    options = {"ents": [ent.label_ for ent in doc.ents], "colors": colors}
    html = displacy.render(doc, style='ent', options=options, page=True, minify=True)
    st.write(html, unsafe_allow_html=True)
    return html

#--------------------------------- Fraud Detection -----------------------------------
@st.cache_data
def ner_list_similarity_jaccard(ner_list1, ner_list2):
    """Calculates Jaccard similarity between two lists of NER tokens after removing stopwords."""
    filtered_ner_list1 = [token for token in ner_list1 if token.lower() not in stop_words]
    filtered_ner_list2 = [token for token in ner_list2 if token.lower() not in stop_words]
    
    intersection_size = len(set(filtered_ner_list1).intersection(filtered_ner_list2))
    union_size = len(set(filtered_ner_list1).union(filtered_ner_list2))
    
    return intersection_size / union_size if union_size else 0

def fraud(detail):
    """Identifies potential fraud by comparing extracted NER with database entries using Jaccard similarity."""
    if not isinstance(detail, spacy.tokens.Doc):
        raise ValueError("Input to `fraud` must be a SpaCy Doc object.")
    
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        st.error("Medicine data file not found.")
        return None, None, "Data file missing"
    
    # Data preparation
    df.fillna(' ', inplace=True)
    
    # Extract entities from SpaCy Doc object
    entity_dict = {label: [] for label in ["sub_category", "product_name", "salt_comp", "manufactured_by"]}
    for ent in detail.ents:
        if ent.label_ in entity_dict:
            entity_dict[ent.label_].append(ent.text)
    
    flattened_list = [item for sublist in entity_dict.values() for item in sublist]
    example_tokens = word_tokenize(' '.join(flattened_list))
    
    max_jaccard_score, max_jaccard_index = -1, -1
    
    # Calculate Jaccard similarity for each row in the DataFrame
    for index, row in df.iterrows():
        base_ner_list = [row[col] for col in ["product_name", "manufactured_by", "salt_comp", "sub_category"]]
        base_tokens = word_tokenize(' '.join(base_ner_list))
        
        # Filter tokens
        filtered_base_tokens = [token for token in base_tokens if token.lower() not in stop_words and token not in [',', '.', ':', 'nan']]
        filtered_example_tokens = [token for token in example_tokens if token.lower() not in stop_words and token not in [',', '.', ':', 'nan']]
        
        # Calculate Jaccard similarity
        jaccard_similarity = ner_list_similarity_jaccard(filtered_base_tokens, filtered_example_tokens)
        
        if jaccard_similarity > max_jaccard_score:
            max_jaccard_score, max_jaccard_index = jaccard_similarity, index

    # Retrieve entity data and set fraud status based on threshold
    threshold = 0.1
    entities = df.loc[max_jaccard_index] if max_jaccard_index != -1 else None
    fraud_status = "This Drug is potentially safe" if max_jaccard_score > threshold else "This Drug is potentially fraudulent"
    
    return entities, max_jaccard_score, fraud_status