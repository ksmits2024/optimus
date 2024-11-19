import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd
import numpy as np
from scipy.special import softmax

# ---------------------------- Caching Models ----------------------------

@st.cache_resource
def load_embedding_model():
    """
    Load and cache the sentence embedding model.
    """
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

@st.cache_resource
def load_sentiment_model():
    """
    Load and cache the sentiment analysis tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
    model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
    return tokenizer, model

@st.cache_resource
def load_classification_pipeline():
    """
    Load and cache the zero-shot classification pipeline.
    """
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return classifier

# ---------------------------- Sentiment Analysis Function ----------------------------

def analyze_sentiment(sentence, tokenizer, model):
    """
    Analyze the sentiment of a given sentence.
    
    Args:
        sentence (str): The sentence to analyze.
        tokenizer: The tokenizer for the sentiment model.
        model: The sentiment analysis model.
        
    Returns:
        sentiment (str): The predicted sentiment label.
        confidence (float): The confidence score for the prediction.
    """
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

# ---------------------------- Main Problem Classification Function ----------------------------

def classify_main_problem(sentence, classifier, candidate_labels):
    """
    Classify the main problem category of a given sentence using zero-shot classification.
    
    Args:
        sentence (str): The sentence to classify.
        classifier: The zero-shot classification pipeline.
        candidate_labels (list): List of candidate labels for classification.
        
    Returns:
        label (str): The predicted main problem category.
        score (float): The confidence score for the prediction.
    """
    result = classifier(sentence, candidate_labels)
    label = result['labels'][0]
    score = result['scores'][0]
    return label, score

# ---------------------------- Streamlit Application ----------------------------

def main():
    st.title("Customer Complaints Processor with Hugging Face Models")
    st.write("""
        Upload a CSV file containing customer complaints. The app will generate sentence embeddings, perform sentiment analysis, and classify the main problem category for each complaint.
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

            # Define candidate labels for main problem classification
            candidate_labels = ['Product', 'Shipping', 'Service', 'Pricing', 'Others']

            # Load models
            with st.spinner('Loading models...'):
                embedding_model = load_embedding_model()
                sentiment_tokenizer, sentiment_model = load_sentiment_model()
                classifier = load_classification_pipeline()

            # Process embeddings
            with st.spinner('Generating sentence embeddings...'):
                embeddings = embedding_model.encode(complaints.tolist(), show_progress_bar=True)

            # Process sentiment analysis and main problem classification
            sentiments = []
            sentiment_confidences = []
            main_problems = []
            problem_confidences = []

            with st.spinner('Analyzing sentiments and classifying main problems...'):
                progress_bar = st.progress(0)  # Initialize progress bar
                total = len(complaints)
                for i, sentence in enumerate(complaints, 1):
                    # Sentiment Analysis
                    sentiment, sentiment_conf = analyze_sentiment(sentence, sentiment_tokenizer, sentiment_model)
                    sentiments.append(sentiment)
                    sentiment_confidences.append(sentiment_conf)

                    # Main Problem Classification
                    problem, problem_conf = classify_main_problem(sentence, classifier, candidate_labels)
                    main_problems.append(problem)
                    problem_confidences.append(problem_conf)

                    # Update progress bar
                    progress = i / total
                    progress_bar.progress(progress)

            # Create results DataFrame
            results_df = pd.DataFrame({
                'Complaint': complaints,
                'Sentiment': sentiments,
                'Sentiment Confidence': [f"{conf:.2%}" for conf in sentiment_confidences],
                'Main Problem': main_problems,
                'Problem Confidence': [f"{conf:.2%}" for conf in problem_confidences],
                # 'Embedding': embeddings.tolist()  # Embeddings are large; handle as needed
            })

            st.subheader("Sentiment Analysis and Problem Classification Results")
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

            # Visualize main problem distribution
            st.subheader("Main Problem Distribution")
            problem_counts = results_df['Main Problem'].value_counts()
            st.bar_chart(problem_counts)

        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty.")
        except pd.errors.ParserError:
            st.error("Error parsing the CSV file. Please ensure it's properly formatted.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

