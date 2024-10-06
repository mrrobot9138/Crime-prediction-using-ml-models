import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

# Initialize the NLTK sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define lexicons for specific emotions
anger_words = set(['angry', 'furious', 'outraged', 'irritated', 'annoyed', 'mad'])
distress_words = set(['distressed', 'upset', 'worried', 'anxious', 'troubled', 'concerned'])
rage_words = set(['rage', 'enraged', 'fuming', 'livid', 'seething', 'irate'])

def analyze_sentiment_and_emotions(text):
    sentiment = sia.polarity_scores(text)
    tokens = word_tokenize(text.lower())
    
    anger_score = len([word for word in tokens if word in anger_words])
    distress_score = len([word for word in tokens if word in distress_words])
    rage_score = len([word for word in tokens if word in rage_words])
    
    return {
        'compound': sentiment['compound'],
        'anger': anger_score,
        'distress': distress_score,
        'rage': rage_score
    }

def categorize_sentiment(scores):
    if scores['compound'] > 0.05:
        return 'Positive'
    elif scores['compound'] < -0.05:
        if scores['anger'] > 0:
            return 'Negative (Anger)'
        elif scores['distress'] > 0:
            return 'Negative (Distress)'
        elif scores['rage'] > 0:
            return 'Negative (Rage)'
        else:
            return 'Negative'
    else:
        return 'Neutral'

def main():
    # Read the CSV file
    df = pd.read_csv('texts.csv')
    
    # Ensure there's a 'text' column in the CSV
    if 'text' not in df.columns:
        print("Error: CSV file must contain a 'text' column")
        return
    
    # Apply sentiment and emotion analysis to each text
    df['analysis'] = df['text'].apply(analyze_sentiment_and_emotions)
    
    # Extract scores and categorize
    df['compound'] = df['analysis'].apply(lambda x: x['compound'])
    df['anger'] = df['analysis'].apply(lambda x: x['anger'])
    df['distress'] = df['analysis'].apply(lambda x: x['distress'])
    df['rage'] = df['analysis'].apply(lambda x: x['rage'])
    df['sentiment_category'] = df['analysis'].apply(categorize_sentiment)
    
    # Print results
    print(df[['text', 'compound', 'anger', 'distress', 'rage', 'sentiment_category']])
    
    # Plot sentiment distribution
    sentiment_counts = df['sentiment_category'].value_counts()
    plt.figure(figsize=(12, 6))
    sentiment_counts.plot(kind='bar')
    plt.title('Sentiment and Emotion Distribution')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('sentiment_emotion_distribution.png')
    print("Sentiment and emotion distribution plot saved as 'sentiment_emotion_distribution.png'")

if __name__ == "__main__":
    main()
