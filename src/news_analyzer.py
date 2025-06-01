from clean_text import TextCleaner
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import pandas as pd
from textblob import TextBlob
import numpy as np
from scipy.stats import pearsonr

class TextAnalyzer:
    def __init__(self, df):
        if isinstance(df, str):
            self.df = pd.read_csv(df)
        else:
            self.df = df
        self.cleaner = TextCleaner()
        self.clean_data()

    def clean_data(self):
        """Clean the dataframe"""
        self.df = self.cleaner.clean_data_frame(self.df, 'headline')
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        # Drop rows with null dates as they're essential for correlation
        self.df = self.df.dropna(subset=['date'])

    def analyze_sentiment(self, text_column='headline'):
        """Analyze sentiment of headlines using TextBlob"""
        self.df['sentiment'] = self.df[text_column].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        self.df['sentiment_category'] = pd.cut(
            self.df['sentiment'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['negative', 'neutral', 'positive']
        )
        return self.df

    def temporal_analysis(self):
        """Analyze temporal patterns with sentiment"""
        self.df['hour'] = self.df['date'].dt.hour
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        self.df['date_only'] = self.df['date'].dt.date
        return self.df[['hour', 'day_of_week', 'sentiment']].describe()

    def publisher_analysis(self):
        """Analyze publisher patterns with sentiment"""
        publisher_stats = self.df.groupby('publisher').agg(
            count=('publisher', 'size'),
            avg_sentiment=('sentiment', 'mean')
        ).reset_index()
        return publisher_stats.sort_values('count', ascending=False)

    def extract_topics(self, n_topics=5, n_words=5):
        """Extract common topics using simple bag-of-words approach"""
        vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=100)
        X = vectorizer.fit_transform(self.df['headline'])
        words = vectorizer.get_feature_names_out()
        word_counts = X.sum(axis=0).A1
        word_freq = dict(zip(words, word_counts))
        
        # Get top n-grams
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:n_topics*n_words]

    def analyze_email_domains(self):
        """Extract email domains from publisher field"""
        email_publishers = self.df[self.df['publisher'].str.contains('@', na=False)]
        email_publishers['domain'] = email_publishers['publisher'].str.extract(r'@(.+)')
        return email_publishers['domain'].value_counts()

    def prepare_correlation_data(self):
        """Prepare data for correlation analysis"""
        if 'sentiment' not in self.df.columns:
            self.analyze_sentiment()
        
        # Group by date and calculate average sentiment
        daily_sentiment = self.df.groupby('date_only').agg(
            avg_sentiment=('sentiment', 'mean'),
            article_count=('sentiment', 'size')
        ).reset_index()
        
        return daily_sentiment