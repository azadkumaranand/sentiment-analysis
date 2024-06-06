from django.shortcuts import render
from django.http import HttpResponse
import os
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Define paths to the model and vectorizer files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'sentiment_model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'vectorizer.pkl')

# Load the trained model and vectorizer
with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub('<br />', ' ', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

def home(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        if text:
            try:
                processed_text = preprocess_text(text)
                text_vector = vectorizer.transform([processed_text]).toarray()
                prediction = model.predict(text_vector)
                sentiment = ''
                if prediction[0] == 1:
                    sentiment = 'positive'
                elif prediction[0] == 2:
                    sentiment = 'neutral'
                else:
                    sentiment = 'negative'
                return render(request, 'result.html', {'sentiment': sentiment})
            except Exception as e:
                return render(request, 'result.html', {'error': str(e)})
        else:
            return render(request, 'result.html', {'error': 'No text provided'})
    return render(request, 'home.html')
