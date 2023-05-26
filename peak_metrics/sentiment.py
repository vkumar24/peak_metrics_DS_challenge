import warnings
from textblob import TextBlob
warnings.filterwarnings("ignore")

class SentimentTrainer:
    def __init__(self):
        pass

    def compute_sentiment_score(self, text):
        sentiment_score = TextBlob(str(text)).sentiment.polarity
        return sentiment_score

    def label_sentiment(self, sentiment_score):
        if sentiment_score > 0:
            sentiment_label = 'Positive'
        elif sentiment_score < 0:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'
        return sentiment_label

    def __call__(self, text):
        sentiment_score = self.compute_sentiment_score(text)
        sentiment_label = self.label_sentiment(sentiment_score)
        return sentiment_score, sentiment_label
