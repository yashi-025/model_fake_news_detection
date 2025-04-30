📰 ***Fake News Detection Model***
A machine learning web app that detects whether a news article is real or fake using Natural Language Processing (NLP) techniques. Built with Python, Scikit-learn, logistic Regression and Flask.

📌 Table of Contents
- Overview
- Prerequisites
- Directory Structure
- Setup and Installation
- Usage
- Model Details
- License

# Overview
The Fake News Detection model is a web-based application built using Flask to create an interactive interface for users to input news text. The application uses a machine learning model trained using Logistic Regression for text classification to classify news as either "Fake" or "Real". The project employs Natural Language Processing (NLP) techniques for feature extraction from news text, and the model is serialized as a .pkl file for inference.

# Prerequisites
Before running the project, ensure you have the following installed:
  - Python (>=3.6)
  - Flask
  - Scikit-learn
  - Pandas
  - Numpy
  - NLTK
  - Pickle

You can install the required Python libraries using:
'pip install -r requirements.txt'

# Directory Structure
The project has the following directory structure:

'''
Fake_News_Detection/
│
├── dataset/                  # Contains raw and pre-processed data for training
│   └── data.csv              # Dataset file used for training
│
├── model/                    # Contains the trained model file
│   └── fake_news_model.pkl   # Serialized Logistic Regression model
│
├── templates/                # HTML templates for the Flask app
│   ├── index.html            # Homepage template
│   └── result.html           # Result display template
│
├── generate.py               # Script to train and save the model
└──  app.py                    # Flask app for serving the model
'''

