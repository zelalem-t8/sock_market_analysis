import pandas as pd
from news_analyzer import TextAnalyzer
from correlation_analyzer import CorrelationAnalyzer
import matplotlib.pyplot as plt
from stock import load_stock_data, calculate_financial_metrics

def main():
    # Load and analyze news data
    news_df = pd.read_csv('../data/raw_analyst_ratings.csv/raw_analyst_ratings.csv')
    analyzer = TextAnalyzer(news_df)
    analyzer.analyze_sentiment()
    
    # Perform analyses
    temporal_stats = analyzer.temporal_analysis()
    publisher_stats = analyzer.publisher_analysis()
    top_topics = analyzer.extract_topics()
    domain_stats = analyzer.analyze_email_domains()
    daily_sentiment = analyzer.prepare_correlation_data()
    
    # Load and analyze stock data
    stock_df = load_stock_data('TSLA_historical_data')  # Example with Tesla
    stock_metrics = calculate_financial_metrics(stock_df)
    stock_df = pd.concat([stock_df, pd.DataFrame(stock_metrics)], axis=1)
    
    # Correlation analysis
    correlator = CorrelationAnalyzer(daily_sentiment, stock_df)
    correlator.align_data()
    correlation_results = correlator.calculate_correlations()
    
    # Visualize results
    visualize_results(
        temporal_stats, 
        publisher_stats, 
        top_topics, 
        domain_stats,
        correlation_results
    )
    correlator.visualize_correlations()
    
def visualize_results(temporal, publishers, topics, domains, correlations):
    """Create visualizations from analysis results"""
    plt.figure(figsize=(30, 10))
    
    # Publisher analysis plot
    plt.subplot(1, 2, 1)
    publishers.head(10).plot.barh(x='publisher', y='count', color='skyblue')
    plt.title('Top Publishers by Article Count', fontsize=8)
    plt.xlabel('Count', fontsize=6)
    plt.ylabel('Publisher', fontsize=6)
    
    # Email domains pie chart
    plt.subplot(1, 2, 2)
    pie_chart = domains.head(5).plot.pie(
        autopct='%1.1f%%', startangle=90, 
        colors=['lightblue', 'lightgreen', 'salmon', 'gold', 'lightcoral']
    )
    plt.title('Email Domain Distribution', fontsize=8)
    plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.show()
    
    # Print topics
    print("\nTop Topics Identified:")
    for i, (word, count) in enumerate(topics, 1):
        print(f"{i}. {word} (Count: {count})")
    
    # Print correlation results
    print("\nCorrelation Results:")
    for metric, result in correlations.items():
        print(f"{metric}: {result['correlation']:.3f} (p-value: {result['p_value']:.3f})")

if __name__ == "__main__":
    main()