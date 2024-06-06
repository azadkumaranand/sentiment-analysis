import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import nltk

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
file_path = 'D:/python/sediment/sentiment_analysis/test_twitter_x_test.csv'
df = pd.read_csv(file_path)

# Define the columns
TEXT_COLUMN = 'text'
SENTIMENT_COLUMN = 'sentiment'

# Check the distribution of sentiments
print(df[SENTIMENT_COLUMN].value_counts())

# Fill missing values in the text column with an empty string
df[TEXT_COLUMN].fillna('', inplace=True)

# Fill missing values in the sentiment column with a placeholder and remove them later
df[SENTIMENT_COLUMN].fillna('unknown', inplace=True)

# Data cleaning and preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub('<br />', ' ', text)  # Remove HTML tags
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Split into words
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

# Apply preprocessing to the dataset
df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(preprocess_text)

# Convert sentiments to numeric labels, handling unknown labels
df = df[df[SENTIMENT_COLUMN] != 'unknown']
df[SENTIMENT_COLUMN] = df[SENTIMENT_COLUMN].map({'positive': 1, 'neutral': 2, 'negative': 0})

# Remove rows with unknown sentiments
df = df.dropna(subset=[SENTIMENT_COLUMN])

# Vectorize the text data using TF-IDF with n-grams
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(df[TEXT_COLUMN]).toarray()

# Use SMOTEENN to balance the dataset
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X_vec, df[SENTIMENT_COLUMN])

# Print class distribution after balancing
print(pd.Series(y_resampled).value_counts())

# Split the resampled data into training and testing sets using stratified split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Train a Voting Classifier with XGBoost and RandomForest
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
rf = RandomForestClassifier()

voting_clf = VotingClassifier(estimators=[
    ('xgb', xgb),
    ('rf', rf)
], voting='soft')

# Perform a randomized search for the best parameters
param_distributions = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 6, 10],
    'xgb__learning_rate': [0.01, 0.1],
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 10, 20]
}

random_search = RandomizedSearchCV(estimator=voting_clf, param_distributions=param_distributions, n_iter=10, cv=5, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train, y_train)

# Best estimator found by randomized search
model = random_search.best_estimator_

# Evaluate the model on the test data
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print(classification_report(y_test, y_pred, target_names=['negative', 'positive', 'neutral']))

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save the trained model and vectorizer
with open('sentiment_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

print("Model and vectorizer saved successfully.")
