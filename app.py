import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from wordcloud import WordCloud
from fpdf import FPDF
import numpy as np
from datetime import datetime
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load Dataset
@st.cache_data
def load_data():
    data = pd.read_csv("drugsComTest_raw.csv")
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.dropna(subset=['review', 'rating', 'date'])  # Drop rows with missing values
    return data

data = load_data()

# Sidebar for User Input
st.sidebar.title("Drug Sentiment Analysis")
selected_drugs = st.sidebar.multiselect(
    "Select Drugs for Analysis",
    options=data["drugName"].unique(),
    default=data["drugName"].unique()[:5]
)

# Filter data
filtered_data = data[data["drugName"].isin(selected_drugs)]

# Sentiment Analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Sentiment Classification
def classify_sentiment(score):
    if score > 0.2:
        return "Positive"
    elif score < -0.2:
        return "Negative"
    else:
        return "Neutral"

# Adding Sentiment Analysis
filtered_data['sentiment_score'] = filtered_data['review'].apply(analyze_sentiment)
filtered_data['sentiment'] = filtered_data['sentiment_score'].apply(classify_sentiment)

# Main Dashboard
st.title("Drug Sentiment Analysis Dashboard")
st.write("Analyze user reviews about drugs and predict their future trends.")

# Sentiment Distribution
st.subheader("Sentiment Distribution")
sentiment_counts = filtered_data['sentiment'].value_counts()
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis", ax=ax)
ax.set_title("Sentiment Distribution")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Count")
st.pyplot(fig)

# Word Clouds
st.subheader("Word Cloud of Reviews")
for drug in selected_drugs:
    st.write(f"**{drug}**")
    reviews = " ".join(filtered_data[filtered_data["drugName"] == drug]["review"].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(reviews)
    st.image(wordcloud.to_array(), width=900)

# Time-Series Prediction using Prophet
st.subheader("Drug Usage Prediction")
drug_usage = (
    filtered_data.groupby(['date', 'drugName']).size().reset_index(name='counts')
)
predictions = {}

for drug in selected_drugs:
    st.write(f"**{drug}**")
    drug_data = drug_usage[drug_usage['drugName'] == drug]
    if not drug_data.empty:
        df_prophet = drug_data.rename(columns={"date": "ds", "counts": "y"})
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        predictions[drug] = forecast
        fig = model.plot(forecast)
        st.pyplot(fig)

# Feature Correlation Analysis
st.subheader("Feature Correlation")
corr_data = filtered_data[['rating', 'sentiment_score', 'usefulCount']].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_data, annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation Heatmap")
st.pyplot(fig)

# PDF Report Generation
def generate_pdf_report(data, selected_drugs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.set_font("Arial", style="B", size=16)  # Bold font for the title
    pdf.cell(200, 10, txt="RxSentinel Drug Analysis Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)

    pdf.ln(10)  # Adds some space

    # Sentiment Analysis
    pdf.cell(200, 10, txt="Sentiment Analysis", ln=True, align="L")
    sentiment_counts = data['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        pdf.cell(200, 10, txt=f"{sentiment}: {count}", ln=True)
    
    pdf.ln(10)

    # Drug-Specific Insights
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="Drug-Specific Insights", ln=True)
    pdf.set_font("Arial", size=12)

    for drug in selected_drugs:
        pdf.set_font("Arial", style="B", size=12)  # Bold drug name
        pdf.cell(200, 10, txt=f"{drug}", ln=True)
        pdf.set_font("Arial", size=12)

        # Filter's data for the drug
        drug_data = data[data["drugName"] == drug]

        # Calculate average rating and sentiment
        avg_rating = drug_data['rating'].mean()
        avg_sentiment = drug_data['sentiment_score'].mean()
        pdf.cell(200, 10, txt=f"Average Rating: {avg_rating:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Average Sentiment: {avg_sentiment:.2f}", ln=True)

        pdf.ln(5)  # Adds some space after each drug

    # Save PDF
    pdf.output("rx_report.pdf")


# Download PDF Report
if st.button("Generate PDF Report"):
    generate_pdf_report(filtered_data, selected_drugs)
    with open("rx_report.pdf", "rb") as file:
        st.download_button(
            label="Download Report",
            data=file,
            file_name="rx_report.pdf",
            mime="application/pdf"
        )
