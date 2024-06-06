# Sentiment Analysis Project

This project is a sentiment analysis system developed using Python, Scikit-learn, and XGBoost. It analyzes textual data to classify sentiments into positive, neutral, or negative categories.

## Project Overview

- **Objective**: To build a sentiment analysis model that can classify text into positive, neutral, or negative sentiments.
- **Technologies Used**: Python, Scikit-learn, XGBoost, NLTK, Django, SMOTEENN.

## Dataset

The dataset used for this project is from Twitter, containing tweets and their corresponding sentiment labels.

## Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/azadkumaranand/sentiment-analysis.git
    cd sentiment-analysis-project
    ```

2. **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download NLTK Data**:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

5. **Run the Training Script**:
    ```bash
    python train_model.py
    ```

6. **Run the Django Application**:
    ```bash
    python manage.py runserver
    ```

## Usage

- **Input**: Enter a text string in the web application to analyze its sentiment.
- **Output**: The model will classify the input text as positive, neutral, or negative.

## Project Structure

