# Fake-Medicine-Detection-App

## Overview
The Fake Medicine Detection App is a machine learning-powered application designed to help users detect counterfeit medicines. Using Optical Character Recognition (OCR), Named Entity Recognition (NER), and Jaccard similarity-based fraud detection, this application allows users to identify potentially fraudulent medicines from their images.

## Features
- User Authentication: Secure login and account creation using bcrypt to store hashed passwords.
- OCR Text Extraction: Extracts text from uploaded images or live camera feed.
- Named Entity Recognition: Identifies key medicine-related entities in the extracted text.
- Fraud Detection: Compares extracted information with a trusted dataset of medicines to flag potential counterfeits.
  
## Table of Contents
- Technologies Used
- Getting Started
- Usage
- File Descriptions
  
## Technologies Used
- Programming Language: Python
- Libraries: Streamlit, EasyOCR, Spacy, OpenCV, bcrypt, MySQL, Pandas, NLTK, and more.
- Database: MySQL (for user credentials)
- ML Models: Custom Spacy NER model, EasyOCR for text extraction

## Getting Started
- Clone the repository:

  ```bash
      git clone https://github.com/Adya-Mishra/Fake-Medicine-Detection-App.git
      cd Fake-Medicine-Detection-App

- Set up a MySQL Database:

1. Create a database called fake_pharma_users.
2. Create a table users with fields email and password for storing user credentials.

- Install Dependencies:

  ```bash
      pip install -r requirements.txt

- Run the App:

  ```bash
      streamlit run login_page.py

## Usage
- Login / Sign Up:
  
Users can create an account and log in to access the app.

- Extract Text from Images:
  
Upload an image or use the live camera feed to capture a medicine image.
Click on "Extract Text from Image" to extract text.

- Display Labeled Text (NER):

After text extraction, perform Named Entity Recognition to identify key entities like product name, manufacturer, etc.

- Fraud Detection:

Based on the extracted entities, perform fraud detection using Jaccard similarity with a trusted medicine dataset.

## File Descriptions
- login_page.py: Manages user authentication (login and sign-up), database connection, and background image settings.
- main_app.py: Main application logic, including sidebar navigation, image upload, OCR text extraction, and NER visualization.
- ner_app.py: Contains functions for image preprocessing, OCR extraction, Named Entity Recognition, and fraud detection based on Jaccard similarity.
