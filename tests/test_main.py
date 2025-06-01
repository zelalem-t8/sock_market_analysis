import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from main import main, visualize_results

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02'],
        'publisher': ['CNN', 'Reuters'],
        'headline': ['Market news today', 'Earnings report'],
        'stock': ['AAPL', 'TSLA']
    })

@pytest.fixture
def mock_stock_data():
    return pd.DataFrame({
        'Close': [150, 155],
        'Volume': [1000000, 2000000],
        'Daily_Return': [0.01, 0.03]
    }, index=pd.to_datetime(['2023-01-01', '2023-01-02']))

def test_main_integration(sample_data, mock_stock_data):
    """Test if main() integrates all components correctly."""
    with patch('pandas.read_csv', return_value=sample_data), \
         patch('main.load_stock_data', return_value=mock_stock_data), \
         patch('main.TextAnalyzer') as MockAnalyzer, \
         patch('main.CorrelationAnalyzer') as MockCorrelator, \
         patch('main.visualize_results') as mock_visualize, \
         patch('main.plt.show'):

        # Setup mock analyzer
        mock_analyzer = MockAnalyzer.return_value
        mock_analyzer.temporal_analysis.return_value = "temporal_data"
        mock_analyzer.publisher_analysis.return_value = pd.DataFrame({
            'publisher': ['CNN', 'Reuters'],
            'count': [1, 1],
            'avg_sentiment': [0.1, 0.2]
        })
        mock_analyzer.extract_topics.return_value = [('stocks', 5)]
        mock_analyzer.analyze_email_domains.return_value = pd.Series({'example.com': 1})
        mock_analyzer.prepare_correlation_data.return_value = pd.DataFrame({
            'date_only': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'avg_sentiment': [0.1, 0.2],
            'article_count': [1, 1]
        })

        # Setup mock correlator
        mock_correlator = MockCorrelator.return_value
        mock_correlator.calculate_correlations.return_value = {
            'pearson_sentiment_returns': {'correlation': 0.5, 'p_value': 0.01}
        }

        main()

        # Verify visualization was called
        mock_visualize.assert_called_once()
        
        # Verify correlation analysis was performed
        MockCorrelator.return_value.visualize_correlations.assert_called_once()

def test_visualize_results():
    """Test the visualize_results function with mock data."""
    with patch('main.plt.subplot'), \
         patch('main.plt.title'), \
         patch('main.plt.xlabel'), \
         patch('main.plt.ylabel'), \
         patch('main.plt.tight_layout'), \
         patch('main.plt.show'):
        
        temporal_stats = pd.DataFrame()
        publisher_stats = pd.DataFrame({
            'publisher': ['CNN', 'Reuters'],
            'count': [10, 8]
        })
        top_topics = [('stocks', 5), ('earnings', 4)]
        domain_stats = pd.Series({'example.com': 5, 'finance.com': 3})
        correlations = {
            'pearson_sentiment_returns': {'correlation': 0.5, 'p_value': 0.01}
        }
        
        visualize_results(temporal_stats, publisher_stats, top_topics, domain_stats, correlations)
        
        # If we get here without errors, the test passes
        assert True