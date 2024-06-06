import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the trained model and vectorizer
with open('sentiment_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub('<br />', ' ', text)  # Remove HTML tags
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Split into words
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

# Sample inputs
samples = [
    "I had a great experience with the customer service.",
    "The product quality is excellent and exceeded my expectations.",
    "I love using this app; it makes my life so much easier.",
    "The movie was fantastic, and the acting was top-notch.",
    "The vacation was wonderful, and we had a great time.",
    "The service was okay, nothing special.",
    "The product is decent for the price.",
    "The app is fine, but it could use some improvements.",
    "The movie was average, not too bad, but not great either.",
    "The vacation was alright, nothing extraordinary.",
    "I am very disappointed with the service I received.",
    "The product broke after just one use. Very poor quality.",
    "This app is terrible; it crashes all the time and is very slow.",
    "The movie was boring, and the acting was subpar.",
    "The vacation was a disaster; nothing went as planned."
]

# Preprocess samples
preprocessed_samples = [preprocess_text(sample) for sample in samples]
for sample, preprocessed in zip(samples, preprocessed_samples):
    print(f"Original: {sample}\nPreprocessed: {preprocessed}\n")

# Transform the preprocessed samples
vectorized_samples = vectorizer.transform(preprocessed_samples).toarray()
print(vectorized_samples)

# Predict sentiment
predictions = model.predict(vectorized_samples)

# Output results
for sample, prediction in zip(samples, predictions):
    if prediction == 1:
        sentiment = 'positive'
    elif prediction == 2:
        sentiment = 'neutral'
    else:
        sentiment = 'negative'
    print(f"Text: {sample}\nSentiment: {sentiment}\n")
