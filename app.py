from flask import Flask, request, render_template, jsonify
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load model and vectorizer
with open('models/fake_news_bundle.pkl', 'rb') as f:
    bundle = pickle.load(f)
    model = bundle['model']
    vectorizer = bundle['vectorizer']

# Text Preprocessing Function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# ✅ Web UI Route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict_text():
    user_input = request.form.get('news_text')
    if user_input:
        processed = preprocess_text(user_input)
        transformed = vectorizer.transform([processed])
        prediction = model.predict(transformed)[0]
        return render_template('result.html', prediction='FAKE' if prediction == 0 else 'REAL')
    return render_template('result.html', prediction="No input provided.")

# ✅ Pure API Route
@app.route('/api/predict', methods=['POST','GET'])
def api_predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" in request'}), 400

    processed = preprocess_text(data['text'])
    transformed = vectorizer.transform([processed])
    prediction = model.predict(transformed)[0]

    return jsonify({
        'input': data['text'],
        'processed': processed,
        'prediction': 'FAKE' if prediction == 0 else 'REAL'
    })

if __name__ == '__main__':
    app.run(debug=True)
