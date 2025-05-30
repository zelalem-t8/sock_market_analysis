# Stock Market News Analysis

This project analyzes stock market news headlines to extract insights such as temporal patterns, publisher trends, topic modeling, and email domain analysis. It is designed for use with Python and pandas, and includes utilities for text cleaning and data visualization.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Text Cleaning:** Removes stopwords and non-alphabetic characters from headlines.
- **Temporal Analysis:** Examines patterns by hour and day of week.
- **Publisher Analysis:** Counts and analyzes news publishers.
- **Topic Extraction:** Finds common topics using bag-of-words.
- **Email Domain Analysis:** Extracts and counts email domains from publisher data.
- **Visualization:** (Optional) Visualizes results for reporting.

---

## Project Structure

```
stocke_market_analysis/
│
├── src/
│   ├── main.py              # Main entry point
│   ├── clean_text.py        # Text cleaning utilities
│   ├── news_analyzer.py     # Core analysis logic
│   └── ...                  # Other modules
│
├── tests/
│   └── test_main.py         # Unit and integration tests
│
├── data/
│   └── news.csv             # (Example) Input data file
│
└── README.md                # Project documentation
```

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd stocke_market_analysis
   ```

2. **Create and activate a virtual environment (recommended):**
   ```sh
   python -m venv week1
   week1\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   *If you don’t have a `requirements.txt`, install manually:*
   ```sh
   pip install pandas scikit-learn pytest
   ```

---

## Usage

1. **Prepare your data:**  
   Place your news data CSV (with columns like `date`, `headline`, `publisher`, etc.) in the `data/` folder.

2. **Run the main script:**
   ```sh
   python src/main.py
   ```

   By default, the script will:
   - Load the data
   - Clean and analyze headlines
   - Output or visualize results

3. **Customize input/output:**  
   Edit `src/main.py` to change file paths or analysis parameters as needed.

---

## Testing

Run all tests using `pytest`:

```sh
pytest tests/
```

This will execute the integration and unit tests in `tests/test_main.py`.

---

## Customization

- **Stopwords:**  
  Edit the stopwords list in `src/clean_text.py` to add/remove words as needed.

- **Analysis Columns:**  
  Change the column names in `src/news_analyzer.py` if your data uses different headers.

- **Visualization:**  
  Implement or modify the `visualize_results` function in `src/main.py` to suit your reporting needs.

---

## Troubleshooting

- **Date Parsing Errors:**  
  If you see errors about date formats, ensure your `date` column is consistently formatted, or use `errors='coerce'` in `pd.to_datetime`.

- **NLTK Stopwords:**  
  If you use NLTK for stopwords and get a download error, either download the resource with:
  ```python
  import nltk
  nltk.download('stopwords')
  ```
  or use the built-in stopwords list in `clean_text.py`.

- **Module Import Errors:**  
  Make sure your working directory is the project root and that `src/` is in your Python path.

---

## License

This project is for educational purposes. Please see `LICENSE` for details if provided.

---

## Contact

For questions or contributions, please open an issue or submit a pull request.