import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    Pipeline
)
import torch
import pandas as pd
import numpy as np
from scipy.special import softmax
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

# ---------------------------- Caching Models with Optimizations ----------------------------

@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    """
    Load and cache the sentence embedding model.
    Utilizes GPU if available for faster processing.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
    return model

@st.cache_resource(show_spinner=False)
def load_sentiment_model() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """
    Load and cache the sentiment analysis tokenizer and model.
    Utilizes GPU if available for faster processing.
    """
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
    model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
    model.to(device)
    return tokenizer, model

@st.cache_resource(show_spinner=False)
def load_classification_pipeline() -> Pipeline:
    """
    Load and cache the zero-shot classification pipeline.
    Utilizes GPU if available for faster processing.
    """
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
    return classifier

@st.cache_resource(show_spinner=False)
def load_qa_pipeline() -> Pipeline:
    """
    Load and cache the Question Answering pipeline.
    Utilizes GPU if available for faster processing.
    """
    device = 0 if torch.cuda.is_available() else -1
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)
    return qa_pipeline

# ---------------------------- Sentiment Analysis Function with Batch Processing ----------------------------

def analyze_sentiment_batch(sentences: List[str], tokenizer: AutoTokenizer, model: AutoModelForSequenceClassification, batch_size: int = 32) -> Tuple[List[str], List[float]]:
    """
    Analyze sentiments for a batch of sentences.
    
    Args:
        sentences (List[str]): List of sentences to analyze.
        tokenizer: Tokenizer for the sentiment model.
        model: Sentiment analysis model.
        batch_size (int): Number of samples per batch.
        
    Returns:
        Tuple[List[str], List[float]]: Sentiment labels and confidence scores.
    """
    sentiments = []
    confidences = []
    labels = ['Negative', 'Neutral', 'Positive']
    
    # Process in batches
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        encoded_input = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model(**encoded_input)
        logits = outputs.logits.cpu().numpy()
        scores = softmax(logits, axis=1)
        batch_sentiments = [labels[np.argmax(score)] for score in scores]
        batch_confidences = [np.max(score) for score in scores]
        sentiments.extend(batch_sentiments)
        confidences.extend(batch_confidences)
        
    return sentiments, confidences

# ---------------------------- Main Problem Classification Function with Batch Processing ----------------------------

def classify_main_problem_batch(sentences: List[str], classifier: Pipeline, candidate_labels: List[str], batch_size: int = 32) -> Tuple[List[str], List[float]]:
    """
    Classify main problems for a batch of sentences using zero-shot classification.
    
    Args:
        sentences (List[str]): List of sentences to classify.
        classifier: Zero-shot classification pipeline.
        candidate_labels (List[str]): List of candidate labels.
        batch_size (int): Number of samples per batch.
        
    Returns:
        Tuple[List[str], List[float]]: Predicted labels and confidence scores.
    """
    labels = []
    scores = []
    
    # Process in batches
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        results = classifier(batch, candidate_labels, multi_label=False)
        for result in results:
            labels.append(result['labels'][0])
            scores.append(result['scores'][0])
    
    return labels, scores

# ---------------------------- Context Preparation Functions ----------------------------

def create_context(complaints: pd.Series) -> str:
    """
    Aggregate all complaints into a single context string.
    """
    return " ".join(complaints.tolist())

def create_context_chunks(complaints: pd.Series, chunk_size: int = 1000) -> List[str]:
    """
    Split complaints into chunks of specified size.
    """
    contexts = []
    current_chunk = ""
    for complaint in complaints:
        if len(current_chunk) + len(complaint) + 1 > chunk_size:
            contexts.append(current_chunk)
            current_chunk = complaint
        else:
            current_chunk += " " + complaint
    if current_chunk:
        contexts.append(current_chunk)
    return contexts

# ---------------------------- QA Interface Functions ----------------------------

def add_qa_interface(qa_pipeline, context):
    st.sidebar.header("❓ Ask a Question")
    user_question = st.sidebar.text_input("Enter your question about the complaints data:")
    
    if st.sidebar.button("Get Answer"):
        if user_question.strip() == "":
            st.sidebar.warning("Please enter a valid question.")
        else:
            with st.spinner('🔍 Processing your question...'):
                answer = qa_pipeline(question=user_question, context=context)
                st.sidebar.markdown(f"**Answer:** {answer['answer']} (Confidence: {answer['score']:.2%})")

def add_qa_interface_with_chunks(qa_pipeline, context_chunks):
    st.sidebar.header("❓ Ask a Question")
    user_question = st.sidebar.text_input("Enter your question about the complaints data:")
    
    if st.sidebar.button("Get Answer"):
        if user_question.strip() == "":
            st.sidebar.warning("Please enter a valid question.")
        else:
            with st.spinner('🔍 Processing your question...'):
                answers = []
                confidences = []
                for chunk in context_chunks:
                    try:
                        answer = qa_pipeline(question=user_question, context=chunk)
                        answers.append(answer['answer'])
                        confidences.append(answer['score'])
                    except:
                        continue
                if answers:
                    # Select the answer with the highest confidence
                    best_idx = np.argmax(confidences)
                    st.sidebar.markdown(f"**Answer:** {answers[best_idx]} (Confidence: {confidences[best_idx]:.2%})")
                else:
                    st.sidebar.info("No answer found for the given question.")

# ---------------------------- Streamlit Application with Optimizations ----------------------------

def main():
    st.set_page_config(page_title="Customer Complaints Processor", layout="wide")
    st.title("🚀 Customer Complaints Processor with Hugging Face Models")
    st.markdown("""
        **Upload a CSV file containing customer complaints.** The app will:
        - Generate sentence embeddings
        - Perform sentiment analysis
        - Classify the main problem category for each complaint
        - **Answer your questions about the complaints data**
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("📂 Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Display exact column names
            with st.expander("🔍 View Exact Column Names", expanded=False):
                st.write([repr(col) for col in df.columns.tolist()])
            
            # Display columns list
            with st.expander("📋 Uploaded CSV Columns", expanded=False):
                st.write(df.columns.tolist())
            
            # Display first few rows for verification
            with st.expander("👀 First 5 Rows of the Uploaded CSV", expanded=False):
                st.dataframe(df.head())
            
            # Check if 'Complaint' column exists
            if 'Complaint' not in df.columns:
                st.error("❌ The uploaded CSV does not contain a 'Complaint' column.")
                st.info("ℹ️ Please ensure your CSV has a header named exactly 'Complaint'.")
                st.stop()
            
            # Extract complaints, ensuring they are strings and non-empty
            complaints = df['Complaint'].astype(str).str.strip()
            complaints = complaints[complaints != ""]
            
            st.success(f"✅ **Total Valid Complaints:** {len(complaints)}")
            
            if len(complaints) == 0:
                st.warning("⚠️ No valid complaints found in the 'Complaint' column.")
                st.stop()
            
            # Define candidate labels for main problem classification
            candidate_labels = ['Product', 'Shipping', 'Service', 'Pricing', 'Others']
            
            # Load models with spinner and concurrency
            with st.spinner('🔄 Loading models...'):
                embedding_model, sentiment_tokenizer, sentiment_model, classifier, qa_pipeline = None, None, None, None, None
                with ThreadPoolExecutor(max_workers=4) as executor:
                    future_embedding = executor.submit(load_embedding_model)
                    future_sentiment = executor.submit(load_sentiment_model)
                    future_classification = executor.submit(load_classification_pipeline)
                    future_qa = executor.submit(load_qa_pipeline)
                    embedding_model = future_embedding.result()
                    sentiment_tokenizer, sentiment_model = future_sentiment.result()
                    classifier = future_classification.result()
                    qa_pipeline = future_qa.result()
            
            # Process embeddings
            with st.spinner('🧠 Generating sentence embeddings...'):
                embeddings = embedding_model.encode(
                    complaints.tolist(),
                    show_progress_bar=True,
                    batch_size=64,
                    convert_to_numpy=True,
                    normalize_embeddings=False
                )
            
            # Process sentiment analysis and main problem classification in parallel
            sentiments, sentiment_confidences, main_problems, problem_confidences = [], [], [], []
            
            with st.spinner('📊 Analyzing sentiments and classifying main problems...'):
                # Determine optimal batch size based on model and data
                batch_size = 64
                
                # Use ThreadPoolExecutor to parallelize sentiment and classification
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_sentiment = executor.submit(
                        analyze_sentiment_batch,
                        complaints.tolist(),
                        sentiment_tokenizer,
                        sentiment_model,
                        batch_size
                    )
                    future_classification = executor.submit(
                        classify_main_problem_batch,
                        complaints.tolist(),
                        classifier,
                        candidate_labels,
                        batch_size
                    )
                    
                    sentiments, sentiment_confidences = future_sentiment.result()
                    main_problems, problem_confidences = future_classification.result()
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'Complaint': complaints,
                'Sentiment': sentiments,
                'Sentiment Confidence': [f"{conf:.2%}" for conf in sentiment_confidences],
                'Main Problem': main_problems,
                'Problem Confidence': [f"{conf:.2%}" for conf in problem_confidences],
                # 'Embedding': list(embeddings)  # Uncomment if embeddings are needed
            })
            
            # Display results
            st.subheader("📈 Sentiment Analysis and Problem Classification Results")
            st.dataframe(results_df)
            
            # Option to download results
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Results as CSV",
                data=csv,
                file_name='processed_complaints.csv',
                mime='text/csv',
            )
            
            # Visualization: Sentiment Distribution
            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = results_df['Sentiment'].value_counts().rename_axis('Sentiment').reset_index(name='Counts')
            st.bar_chart(sentiment_counts.set_index('Sentiment'))
            
            # Visualization: Main Problem Distribution
            st.subheader("📦 Main Problem Distribution")
            problem_counts = results_df['Main Problem'].value_counts().rename_axis('Main Problem').reset_index(name='Counts')
            st.bar_chart(problem_counts.set_index('Main Problem'))
            
            # Prepare context for QA
            context = create_context(complaints)
            # For larger datasets, consider using chunked context:
            # context_chunks = create_context_chunks(complaints)
            
            # Add QA Interface
            add_qa_interface(qa_pipeline, context)
            # For chunked context, use:
            # add_qa_interface_with_chunks(qa_pipeline, context_chunks)
        
        except pd.errors.EmptyDataError:
            st.error("❌ The uploaded CSV file is empty.")
        except pd.errors.ParserError:
            st.error("❌ Error parsing the CSV file. Please ensure it's properly formatted.")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
