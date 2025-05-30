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
    # Increase the figure size for better visibility
    plt.figure(figsize=(30, 10))  # Further adjusted size
    
    # Temporal analysis plot
    plt.subplot(1, 2, 1)
    publishers.head(10).plot.barh(x='publisher', y='count', color='skyblue')
    plt.title('Top Publishers by Article Count', fontsize=8)
    plt.xlabel('Count', fontsize=6)
    plt.ylabel('Publisher', fontsize=6)
    
    plt.subplot(1, 2, 2)
    pie_chart = domains.head(5).plot.pie(
        autopct='%1.1f%%', startangle=90, 
        colors=['lightblue', 'lightgreen', 'salmon', 'gold', 'lightcoral']
    )
    plt.title('Email Domain Distribution', fontsize=8)
    plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))  # Moved legend outside the pie chart
    plt.tight_layout()
    plt.show()
    
    # Print topics
    print("\nTop Topics Identified:")
    for i, (word, count) in enumerate(topics, 1):
        print(f"{i}. {word} (Count: {count})")
if __name__ == "__main__":
    main()