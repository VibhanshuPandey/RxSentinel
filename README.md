# RxSentinel: Drug Sentiment Analysis Dashboard

RxSentinel is a Streamlit-based web application that analyzes drug reviews to provide insights into user sentiment, trends, and effectiveness. It uses natural language processing (NLP) and machine learning techniques to classify sentiment, generate word clouds, and predict future trends in drug reviews.

---

## Features

1. **Sentiment Analysis**:
   - Classifies drug reviews as **Positive**, **Neutral**, or **Negative** using TextBlob.
   - Visualizes sentiment distribution using bar charts.

2. **Word Clouds**:
   - Generates word clouds for selected drugs to highlight frequently mentioned terms in reviews.

3. **Time-Series Analysis**:
   - Uses Facebook's Prophet model to predict future trends in drug reviews.

4. **Feature Correlation**:
   - Displays a heatmap showing correlations between features like rating, sentiment score, and usefulness count.

5. **PDF Report Generation**:
   - Generates a downloadable PDF report summarizing key insights, including sentiment analysis and drug-specific metrics.

---

## Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/VibhanshuPandey/RxSentinel.git
   cd RxSentinel
   ```

2. Create a virtual environment:
   ```bash
   python -m venv rxenv
   ```

3. Activate the virtual environment:
   - **Windows**:
     ```bash
     rxenv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source rxenv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Download NLTK data:
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
   ```

---

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the URL provided in the terminal (usually `http://localhost:8501`).

3. Use the sidebar to:
   - Select drugs for analysis.
   - View sentiment distribution, word clouds, and time-series predictions.

4. Generate a PDF report:
   - Click the **Generate PDF Report** button.
   - Download the report using the **Download Report** button.

## Dependencies

- **Streamlit**: For building the web app.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib/Seaborn**: For data visualization.
- **TextBlob**: For sentiment analysis.
- **WordCloud**: For generating word clouds.
- **Prophet**: For time-series forecasting.
- **FPDF**: For generating PDF reports.
- **NLTK**: For natural language processing tasks.

---

## Dataset

The dataset used in this project is `drugsComTest_raw.csv`, which contains drug reviews with the following columns:
- `drugName`: Name of the drug.
- `condition`: Medical condition being treated.
- `review`: User review text.
- `rating`: User rating (out of 10).
- `date`: Date of the review.
- `usefulCount`: Number of users who found the review useful.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- Dataset: [Kaggle Drug Review Dataset](https://www.kaggle.com/jessicali9530/kuc-hackathon-winter-2018)

---
