import pandas as pd
from news_analyzer import TextAnalyzer
import matplotlib.pyplot as plt

def main():
    # Load data
    df = pd.read_csv('./data/raw_analyst_ratings.csv/raw_analyst_ratings.csv')
    
    # Initialize analyzer
    analyzer = TextAnalyzer(df)
    
    # Perform analyses
    temporal_stats = analyzer.temporal_analysis()
    publisher_stats = analyzer.publisher_analysis()
    top_topics = analyzer.extract_topics()
    domain_stats = analyzer.analyze_email_domains()
    
    # Visualize results
    visualize_results(temporal_stats, publisher_stats, top_topics, domain_stats)
    
def visualize_results(temporal, publishers, topics, domains):
    """Create visualizations from analysis results"""
    # Temporal analysis plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    publishers.head(10).plot.barh(x='publisher', y='count')
    plt.title('Top Publishers by Article Count')
    
    plt.subplot(1, 2, 2)
    domains.head(5).plot.pie(autopct='%1.1f%%')
    plt.title('Email Domain Distribution')
    plt.tight_layout()
    plt.show()
    
    # Print topics
    print("\nTop Topics Identified:")
    for i, (word, count) in enumerate(topics, 1):
        print(f"{i}. {word} (Count: {count})")

if __name__ == "__main__":
    main()