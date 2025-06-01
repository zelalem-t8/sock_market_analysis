import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

class TextCleaner:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english')).union({
            'vs', 'said', 'says', 'company', 'companies', 'stock', 'stocks',
            'market', 'markets', 'share', 'shares', 'inc', 'corp', 'llc'
        })
        self.lemmatizer = WordNetLemmatizer()
        self.financial_terms = set([
            'price', 'target', 'upgrade', 'downgrade', 'earnings', 'report',
            'quarter', 'analyst', 'rating', 'buy', 'sell', 'hold', 'forecast'
        ])

    def clean_text(self, text: str):
        text = text.lower()
        # Keep numbers and $ symbols which are important for financial text
        text = re.sub(r'[^a-zA-Z0-9\s$%]', '', text)
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                 if word not in self.stop_words or word in self.financial_terms]
        return ' '.join(tokens)

    def clean_data_frame(self, df, text_column='headline'):
        df[text_column] = df[text_column].apply(self.clean_text)
        return df