import pandas as pd
import re
import os
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
news_dataset = pd.read_csv('dataset/data.csv')
news_dataset = news_dataset.fillna('')

# Preprocessing function
def lemmatizing(content):
    lemmatizer = WordNetLemmatizer()
    content = re.sub('[^a-zA-Z]', ' ', content).lower().split()
    return ' '.join([lemmatizer.lemmatize(word) for word in content if word not in stopwords.words('english')])

# Apply preprocessing
news_dataset['Headline'] = news_dataset['Headline'].apply(lemmatizing)

# Feature and label split
X = news_dataset['Headline'].values
Y = news_dataset['Label'].values

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Save model and vectorizer
os.makedirs('fake_news/models', exist_ok=True)
with open('models/fake_news_bundle.pkl', 'wb') as f:
    pickle.dump({'model': model, 'vectorizer': vectorizer}, f)

print("✅ Model and vectorizer saved to 'models/fake_news_bundle.pkl'")
