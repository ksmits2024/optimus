import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
from scipy.special import softmax

# Caching the embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Caching the sentiment analysis model and tokenizer
@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
    model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
    return tokenizer, model

# Function to analyze sentiment
def analyze_sentiment(sentence, tokenizer, model):
    # Encode the sentence
    encoded_input = tokenizer(sentence, return_tensors='pt', truncation=True)
    with torch.no_grad():
        output = model(**encoded_input)
    scores = output.logits[0].numpy()
    scores = softmax(scores)
    # Get labels (assuming labels are in a specific order)
    labels = ['Negative', 'Neutral', 'Positive']
    sentiment = labels[np.argmax(scores)]
    confidence = np.max(scores)
    return sentiment, confidence

def main():
    st.title("Customer Complaints Processor with Hugging Face Models")
    st.write("""
        Upload a CSV file containing customer complaints. The app will generate sentence embeddings and perform sentiment analysis on each complaint.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)

            # Display exact column names
            st.subheader("Exact Column Names")
            st.write([repr(col) for col in df.columns.tolist()])

            # Display columns list
            st.subheader("Uploaded CSV Columns")
            st.write(df.columns.tolist())

            # Display first few rows for verification
            st.subheader("First 5 Rows of the Uploaded CSV")
            st.write(df.head())

            # Check if 'Complaint' column exists
            if 'Complaint' not in df.columns:
                st.error("The uploaded CSV does not contain a 'Complaint' column.")
                st.info("Please ensure your CSV has a header named exactly 'Complaint'.")
                st.stop()

            # Extract complaints, ensuring they are strings
            complaints = df['Complaint'].astype(str).str.strip()
            complaints = complaints[complaints != ""]  # Remove empty strings

            st.write(f"**Total Complaints:** {len(complaints)}")

            if len(complaints) == 0:
                st.warning("No valid complaints found in the 'Complaint' column.")
                st.stop()

            # Load models
            embedding_model = load_embedding_model()
            tokenizer, sentiment_model = load_sentiment_model()

            # Process embeddings
            with st.spinner('Generating sentence embeddings...'):
                embeddings = embedding_model.encode(complaints.tolist(), show_progress_bar=True)

            # Process sentiment analysis
            sentiments = []
            confidences = []
            with st.spinner('Analyzing sentiments...'):
                for sentence in st.progress(complaints):
                    sentiment, confidence = analyze_sentiment(sentence, tokenizer, sentiment_model)
                    sentiments.append(sentiment)
                    confidences.append(confidence)

            # Create results DataFrame
            results_df = pd.DataFrame({
                'Complaint': complaints,
                'Sentiment': sentiments,
                'Confidence': confidences
                # 'Embedding': embeddings.tolist()  # Embeddings are large; handle as needed
            })

            st.subheader("Sentiment Analysis Results")
            st.dataframe(results_df)

            # Option to download results
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='processed_complaints.csv',
                mime='text/csv',
            )

            # Visualize sentiment distribution
            st.subheader("Sentiment Distribution")
            sentiment_counts = results_df['Sentiment'].value_counts()
            st.bar_chart(sentiment_counts)

        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty.")
        except pd.errors.ParserError:
            st.error("Error parsing the CSV file. Please ensure it's properly formatted.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
