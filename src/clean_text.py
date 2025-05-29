import pandas as pd
import re

class TextCleaner:
    def __init__(self):
        # A minimal English stopwords list
        self.stop_words = set([
            'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on', 'with', 'as', 'by', 'at', 'from', 'it', 'an', 'be'
        ])

    def clean_text(self, text: str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def clean_data_frame(self, df, text_column='headline'):
        df[text_column] = df[text_column].apply(self.clean_text)
        return df