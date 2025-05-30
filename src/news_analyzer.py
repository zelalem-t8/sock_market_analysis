from clean_text import TextCleaner
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import pandas as pd
class TextAnalyzer:
    def __init__(self, df):
        if isinstance(df, str):
            self.df = pd.read_csv(df)
        else:
            self.df = df
        self.cleaner = TextCleaner()
        self.clean_data()
    def clean_data(self):
        """clean the dataframe """
        self.df = self.cleaner.clean_data_frame(self.df, 'headline')
        # Let pandas infer the format and handle missing/inconsistent values
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
    def temporal_analysis(self):
        """Analyze temporal patterns"""
        self.df['hour'] = self.df['date'].dt.hour
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        return self.df[['hour', 'day_of_week']].describe()
    
    def publisher_analysis(self):
        """Analyze publisher patterns"""
        publisher_stats = self.df['publisher'].value_counts().reset_index()
        publisher_stats.columns = ['publisher', 'count']
        return publisher_stats
    
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
        email_publishers = self.df[self.df['publisher'].str.contains('@')]
        email_publishers['domain'] = email_publishers['publisher'].str.extract(r'@(.+)')
        return email_publishers['domain'].value_counts()

