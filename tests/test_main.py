import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from ..src.main import main, visualize_results  # Relative import# Sample test data
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
    with patch('pandas.read_csv', return_value=sample_data) as mock_read_csv, \
         patch('src.main.TextAnalyzer') as MockAnalyzer, \
         patch('src.main.visualize_results') as mock_visualize:

        # Mock analyzer methods
        mock_analyzer = MockAnalyzer.return_value
        mock_analyzer.temporal_analysis.return_value = "temporal_data"
        mock_analyzer.publisher_analysis.return_value = "publisher_data"
        mock_analyzer.extract_topics.return_value = [('stocks', 5)]
        mock_analyzer.analyze_email_domains.return_value = "domain_data"

        # Execute main()
        main()

        # Assertions
        mock_read_csv.assert_called_once_with('./data/raw_analyst_ratings.csv/raw_analyst_ratings.csv')
        MockAnalyzer.assert_called_once_with(sample_data)
        mock_visualize.assert_called_once_with(
            "temporal_data", "publisher_data", [('stocks', 5)], "domain_data"
        )

def test_visualize_results(capsys):
    """Test visualization function with mock data."""
    mock_publishers = pd.DataFrame({
        'publisher': ['CNN', 'Reuters'],
        'count': [10, 5]
    })
    mock_domains = pd.DataFrame({
        'domain': ['gmail.com', 'yahoo.com'],
        'count': [8, 2]
    })

    with patch('matplotlib.pyplot.figure'), \
         patch('matplotlib.pyplot.subplot'), \
         patch('matplotlib.pyplot.show'):
        
        visualize_results(
            temporal=pd.DataFrame(),
            publishers=mock_publishers,
            topics=[('finance', 10)],
            domains=mock_domains
        )

        # Check printed topics
        captured = capsys.readouterr()
        assert "Top Topics Identified:" in captured.out
        assert "1. finance (Count: 10)" in captured.out

def test_empty_data_handling(capsys):
    """Test if function handles empty data gracefully."""
    with patch('matplotlib.pyplot.figure'), \
         patch('matplotlib.pyplot.subplot'), \
         patch('matplotlib.pyplot.show'):
        
        visualize_results(
            temporal=pd.DataFrame(),
            publishers=pd.DataFrame(),
            topics=[],
            domains=pd.DataFrame()
        )

        captured = capsys.readouterr()
        assert "Top Topics Identified:" in captured.out
        assert "1." not in captured.out  # No topics printed