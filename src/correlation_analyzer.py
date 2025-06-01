import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

class CorrelationAnalyzer:
    def __init__(self, news_data, stock_data):
        """
        news_data: DataFrame from TextAnalyzer.prepare_correlation_data()
        stock_data: DataFrame from stock.py with daily returns
        """
        self.news = news_data
        self.stock = stock_data
        
    def align_data(self):
        """Align news and stock data by date"""
        # Ensure both DataFrames have date columns
        self.news['date_only'] = pd.to_datetime(self.news['date_only'])
        self.stock['date_only'] = pd.to_datetime(self.stock.index.date)
        
        # Merge data
        self.merged_data = pd.merge(
            self.news,
            self.stock,
            on='date_only',
            how='inner'
        )
        return self.merged_data
    
    def calculate_correlations(self):
        """Calculate various correlation metrics"""
        if not hasattr(self, 'merged_data'):
            self.align_data()
            
        results = {}
        
        # Pearson correlation between sentiment and daily returns
        corr, p_value = pearsonr(
            self.merged_data['avg_sentiment'],
            self.merged_data['Daily_Return']
        )
        results['pearson_sentiment_returns'] = {
            'correlation': corr,
            'p_value': p_value
        }
        
        # Correlation between sentiment and volume
        corr, p_value = pearsonr(
            self.merged_data['avg_sentiment'],
            self.merged_data['Volume']
        )
        results['pearson_sentiment_volume'] = {
            'correlation': corr,
            'p_value': p_value
        }
        
        return results
    
    def visualize_correlations(self):
        """Create visualizations of correlations"""
        if not hasattr(self, 'merged_data'):
            self.align_data()
            
        plt.figure(figsize=(15, 10))
        
        # Scatter plot: Sentiment vs Returns
        plt.subplot(2, 2, 1)
        sns.regplot(
            x='avg_sentiment',
            y='Daily_Return',
            data=self.merged_data,
            scatter_kws={'alpha':0.3}
        )
        plt.title('Sentiment vs Daily Returns')
        plt.xlabel('Average Daily Sentiment')
        plt.ylabel('Daily Return (%)')
        
        # Sentiment distribution by return sign
        plt.subplot(2, 2, 2)
        self.merged_data['return_positive'] = self.merged_data['Daily_Return'] > 0
        sns.boxplot(
            x='return_positive',
            y='avg_sentiment',
            data=self.merged_data
        )
        plt.title('Sentiment Distribution by Return Sign')
        plt.xlabel('Positive Return')
        plt.ylabel('Average Sentiment')
        plt.xticks([0, 1], ['Negative', 'Positive'])
        
        # Sentiment time series with returns
        plt.subplot(2, 1, 2)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        ax1.plot(
            self.merged_data['date_only'],
            self.merged_data['avg_sentiment'],
            color='blue',
            label='Sentiment'
        )
        ax2.plot(
            self.merged_data['date_only'],
            self.merged_data['Close'],
            color='green',
            label='Price'
        )
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sentiment', color='blue')
        ax2.set_ylabel('Price', color='green')
        plt.title('Sentiment and Price Over Time')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()