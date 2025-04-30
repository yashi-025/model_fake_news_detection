# 📰 *Fake News Detection Model*
A machine learning web app that detects whether a news article is real or fake using Natural Language Processing (NLP) techniques. Built with Python, Scikit-learn, logistic Regression and Flask.

# 📌 Table of Contents
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
`pip install -r requirements.txt`

# Directory Structure
The project has the following directory structure:
```
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
```

# Setup and Installation
1) **Clone the repository:**
```
git clone https://github.com/your-username/FakeNewsDetection.git
cd FakeNewsDetection
```

2) **Train the Model** (optional, if you need to retrain):
Run the generate.py script to train the model and save it as fake_news_model.pkl.
`python generate.py`
The model will be saved in the model/ directory.


# Model Details
The Fake News Detection model is based on Logistic Regression and trained on a text dataset. The model processes text data by performing the following steps:
1) **Text Preprocessing:** Text is cleaned and tokenized. Stopwords are removed, and stemming or lemmatization is applied.
2) **Feature Extraction:** The text data is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency).
3) **Model Training:** Logistic Regression is used to train the model to classify the text as real or fake.

# Files Involved:
- `generate.py`: Contains the code for preprocessing the dataset, vectorizing the text, training the Logistic Regression model, and saving the trained model to a .pkl file.
- `app.py`: The Flask application that loads the trained model and serves the web interface for users to input text and get predictions.
- `model/fake_news_model.pkl`: The trained model file.
- `dataset/data.csv`: The dataset used for training the model.
- `templates/index.html`: The homepage template with a text input box for the user to enter news articles.
- `templates/result.html`: The template to display the classification result (Fake or Real).





