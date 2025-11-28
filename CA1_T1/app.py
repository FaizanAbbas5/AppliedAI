# -*- coding: utf-8 -*-
"""Streamlit app for News Bias Classification"""

import streamlit as st
import joblib
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from news_fetcher import NewsFetcher

# Download NLTK data if not already present
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')

download_nltk_data()

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove special chars/digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stop words and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(cleaned_tokens)

@st.cache_resource
def load_models():
    model_dir = './models'
    
    try:
        models = {
            'Logistic Regression': joblib.load(os.path.join(model_dir, 'logistic_regression.pkl')),
            'Random Forest': joblib.load(os.path.join(model_dir, 'random_forest.pkl')),
            'SVM': joblib.load(os.path.join(model_dir, 'svm.pkl')),
            'XGBoost': joblib.load(os.path.join(model_dir, 'xgboost.pkl'))
        }
        tfidf = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
        label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        
        return models, tfidf, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run the training script (appliedai_t1g.py) first to generate the models.")
        return None, None, None

def get_bias_color(bias):
    """Return color for bias label"""
    colors = {
        'left': '#3498db',    # Blue
        'center': '#2ecc71',  # Green
        'right': '#e74c3c'    # Red
    }
    return colors.get(bias.lower(), '#95a5a6')

def classify_article(content, models, tfidf, label_encoder, selected_model):
    # Preprocess the content
    cleaned_content = preprocess_text(content)
    
    # Transform using TF-IDF
    content_tfidf = tfidf.transform([cleaned_content])
    
    # Get the selected model
    model = models[selected_model]
    
    # Make prediction
    if selected_model == 'XGBoost':
        prediction_enc = model.predict(content_tfidf)
        prediction = label_encoder.inverse_transform(prediction_enc)[0]
    else:
        prediction = model.predict(content_tfidf)[0]
    
    return prediction

def main():
    st.set_page_config(
        page_title="News Bias Classifier",
        page_icon="📰",
        layout="wide"
    )
    
    # Title and description
    st.title("📰 Political News Bias Classifier")
    st.markdown("""
    Search for political news articles and classify their bias in real-time using machine learning models.
    Articles are classified as **Left**, **Center**, or **Right** leaning.
    """)
    
    # Load models
    models, tfidf, label_encoder = load_models()
    
    if models is None:
        st.stop()
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select Classification Model:",
        ['Logistic Regression', 'Random Forest', 'SVM', 'XGBoost'],
        index=0
    )
    
    # API Key input (required)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### News API Configuration")
    st.sidebar.markdown("⚠️ **API Key Required** - Get a free key from [NewsAPI.org](https://newsapi.org/)")
    api_key = st.sidebar.text_input("News API Key (Required):", type="password", help="Enter your NewsAPI.org API key")
    
    if not api_key:
        st.sidebar.error("⚠️ Please enter your News API key to search for articles")
    
    # Main content
    st.markdown("---")
    
    # Search section
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "🔍 Search for political news:",
            placeholder="e.g., healthcare, immigration, climate change...",
            help="Enter keywords to search for political news articles"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    # Search and display results
    if search_button and search_query:
        if not api_key:
            st.error("⚠️ Please enter your News API key in the sidebar to search for articles.")
        else:
            with st.spinner("Fetching and analyzing news articles..."):
                try:
                    # Initialize news fetcher
                    news_fetcher = NewsFetcher(api_key=api_key)
                    
                    # Fetch articles
                    articles = news_fetcher.search_news(search_query, max_results=5)
                    
                    if not articles:
                        st.warning("No articles found. Try a different search query.")
                    else:
                        st.success(f"Found {len(articles)} articles. Classifying with {model_choice}...")
                        
                        # Display articles with classifications
                        for idx, article in enumerate(articles, 1):
                            # Classify the article
                            bias = classify_article(
                                article['content'], 
                                models, 
                                tfidf, 
                                label_encoder,
                                model_choice
                            )
                            
                            # Create expandable card for each article
                            with st.expander(f"📄 Article {idx}: {article['title']}", expanded=(idx == 1)):
                                # Bias classification badge
                                col_bias, col_source = st.columns([1, 2])
                                with col_bias:
                                    bias_color = get_bias_color(bias)
                                    st.markdown(
                                        f"<div style='background-color: {bias_color}; color: white; padding: 10px; "
                                        f"border-radius: 5px; text-align: center; font-weight: bold; font-size: 16px;'>"
                                        f"BIAS: {bias.upper()}</div>",
                                        unsafe_allow_html=True
                                    )
                                with col_source:
                                    st.markdown(f"**Source:** {article['source']}")
                                    st.markdown(f"**Author:** {article['author']}")
                                    st.markdown(f"**Published:** {article['published_at'][:10]}")
                                
                                st.markdown("---")
                                
                                # Article content
                                st.markdown(f"**Title:** {article['title']}")
                                st.markdown(f"**Content Preview:**")
                                
                                # Show first 500 characters of content
                                content_preview = article['content'][:500] + "..." if len(article['content']) > 500 else article['content']
                                st.write(content_preview)
                                
                                # Link to full article
                                if article['url']:
                                    st.markdown(f"[🔗 Read Full Article]({article['url']})")
                                
                                st.markdown("")
                        
                except Exception as e:
                    st.error(f"❌ Error fetching articles: {str(e)}")
                    st.info("Please check your API key and internet connection.")
    
    elif search_button and not search_query:
        st.warning("Please enter a search query.")
    
    # Instructions when no search has been made
    if not search_button:
        st.info("👆 Enter a search query above to find and classify political news articles.")
        
        # Show example
        st.markdown("---")
        st.markdown("### 💡 Example Queries")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.code("healthcare reform")
        with col2:
            st.code("immigration policy")
        with col3:
            st.code("climate change")

if __name__ == "__main__":
    main()
