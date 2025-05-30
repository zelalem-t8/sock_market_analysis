import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from main import main, visualize_results  # Now this will work

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02'],
        'publisher': ['CNN', 'Reuters'],
        'text': ['Market news today', 'Earnings report'],
        'email': ['user@example.com', 'analyst@finance.com']
    })

def test_main_integration(sample_data):
    """Test if main() integrates all components correctly."""
    with patch('pandas.read_csv', return_value=sample_data), \
         patch('main.TextAnalyzer') as MockAnalyzer, \
         patch('main.visualize_results') as mock_visualize:

        mock_analyzer = MockAnalyzer.return_value
        mock_analyzer.temporal_analysis.return_value = "temporal_data"
        mock_analyzer.publisher_analysis.return_value = "publisher_data"
        mock_analyzer.extract_topics.return_value = [('stocks', 5)]
        mock_analyzer.analyze_email_domains.return_value = "domain_data"

        main()

        mock_visualize.assert_called_once_with(
            "temporal_data", "publisher_data", [('stocks', 5)], "domain_data"
        )