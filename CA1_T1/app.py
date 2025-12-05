# -*- coding: utf-8 -*-
"""Streamlit app for News Bias Classification"""

import streamlit as st
import joblib
import os
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from news_fetcher import NewsFetcher
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer

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

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
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
    colors = {
        'left': '#3498db',    
        'center': '#2ecc71',  
        'right': '#e74c3c'  
    }
    return colors.get(bias.lower(), '#95a5a6')

def classify_article(content, models, tfidf, label_encoder, selected_model):
    # Preprocess the content
    cleaned_content = preprocess_text(content)
    
    content_tfidf = tfidf.transform([cleaned_content])
    
    model = models[selected_model]
    
    if selected_model == 'XGBoost':
        prediction_enc = model.predict(content_tfidf)
        prediction = label_encoder.inverse_transform(prediction_enc)[0]
        
        probabilities = model.predict_proba(content_tfidf)[0]
        confidence_scores = {
            label: float(prob) 
            for label, prob in zip(label_encoder.classes_, probabilities)
        }
    else:
        prediction = model.predict(content_tfidf)[0]
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(content_tfidf)[0]
            confidence_scores = {
                label: float(prob) 
                for label, prob in zip(model.classes_, probabilities)
            }
        elif hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(content_tfidf)[0]
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            probabilities = exp_scores / exp_scores.sum()
            confidence_scores = {
                label: float(prob) 
                for label, prob in zip(model.classes_, probabilities)
            }
        else:
            confidence_scores = {}
    
    # Get the confidence for the predicted class
    confidence = confidence_scores.get(prediction, 0.0)
    
    return prediction, confidence, confidence_scores

def get_top_features(content, models, tfidf, label_encoder, selected_model, prediction, top_n=10):
    """Extract top features that influenced the prediction"""
    # Preprocess the content
    cleaned_content = preprocess_text(content)
    
    # Transform using TF-IDF
    content_tfidf = tfidf.transform([cleaned_content])
    
    # Get the selected model
    model = models[selected_model]
    
    # Get feature names
    feature_names = np.array(tfidf.get_feature_names_out())
    
    # Get the non-zero features (words present in this document)
    non_zero_indices = content_tfidf.nonzero()[1]
    
    if len(non_zero_indices) == 0:
        return []
    
    tfidf_scores = content_tfidf.toarray()[0][non_zero_indices]
    
    try:
        if selected_model == 'Logistic Regression':
            if hasattr(model, 'classes_'):
                class_idx = np.where(model.classes_ == prediction)[0][0]
                if len(model.coef_.shape) > 1:
                    coefficients = model.coef_[class_idx]
                else:
                    coefficients = model.coef_[0]
                
                feature_importance = np.abs(coefficients[non_zero_indices]) * tfidf_scores
            else:
                feature_importance = tfidf_scores
                
        elif selected_model == 'Random Forest':
            feature_importance = model.feature_importances_[non_zero_indices] * tfidf_scores
            
        elif selected_model == 'SVM':
            if hasattr(model, 'coef_'):
                class_idx = np.where(model.classes_ == prediction)[0][0]
                if len(model.coef_.shape) > 1:
                    coefficients = model.coef_[class_idx]
                else:
                    coefficients = model.coef_[0]
                feature_importance = np.abs(coefficients[non_zero_indices]) * tfidf_scores
            else:
                feature_importance = tfidf_scores
                
        elif selected_model == 'XGBoost':
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_[non_zero_indices] * tfidf_scores
            else:
                feature_importance = tfidf_scores
        else:
            feature_importance = tfidf_scores
            
        top_indices = np.argsort(feature_importance)[-top_n:][::-1]
        top_features = [
            (feature_names[non_zero_indices[idx]], float(feature_importance[idx]))
            for idx in top_indices
        ]
        
        return top_features
    except Exception as e:
        # Fallback to just TF-IDF scores
        top_indices = np.argsort(tfidf_scores)[-top_n:][::-1]
        top_features = [
            (feature_names[non_zero_indices[idx]], float(tfidf_scores[idx]))
            for idx in top_indices
        ]
        return top_features

def explain_with_lime(content, models, tfidf, label_encoder, selected_model, num_features=10):
    """Use LIME to explain the prediction"""
    model = models[selected_model]
    
    def predict_proba_fn(texts):
        cleaned_texts = [preprocess_text(text) for text in texts]
        text_tfidf = tfidf.transform(cleaned_texts)
        
        if selected_model == 'XGBoost':
            predictions = model.predict_proba(text_tfidf)
        elif hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(text_tfidf)
        elif hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(text_tfidf)
            if len(decision_scores.shape) == 1:
                decision_scores = decision_scores.reshape(-1, 1)
            exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
            predictions = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        else:
            raise ValueError("Model doesn't support probability prediction")
        
        return predictions
    
    if selected_model == 'XGBoost':
        class_names = label_encoder.classes_.tolist()
    else:
        class_names = model.classes_.tolist()
    
    explainer = LimeTextExplainer(class_names=class_names)
    
    exp = explainer.explain_instance(
        content,
        predict_proba_fn,
        num_features=num_features,
        top_labels=3
    )
    
    return exp

def visualize_lime_explanation(exp, prediction):
    """Create a visualization of LIME explanation"""
    # Get explanation for the predicted class
    explanation_list = exp.as_list(label=prediction)
    
    if not explanation_list:
        return None
    
    # Create bar chart
    words = [item[0] for item in explanation_list]
    weights = [item[1] for item in explanation_list]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71' if w > 0 else '#e74c3c' for w in weights]
    
    y_pos = np.arange(len(words))
    ax.barh(y_pos, weights, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Weight')
    ax.set_title(f'Top Features Influencing "{prediction.upper()}" Prediction')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig

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
    st.sidebar.header(" Configuration")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Select Mode:",
        ["Single Model", "Compare Models"],
        help="Choose to use one model or compare two models side-by-side"
    )
    
    # Model selection based on mode
    if mode == "Single Model":
        model_choice = st.sidebar.selectbox(
            "Select Classification Model:",
            ['Logistic Regression', 'Random Forest', 'SVM', 'XGBoost'],
            index=0
        )
        model_choice_2 = None
    else:
        st.sidebar.markdown("### Select Two Models to Compare")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            model_choice = st.sidebar.selectbox(
                "Model 1:",
                ['Logistic Regression', 'Random Forest', 'SVM', 'XGBoost'],
                index=0,
                key="model1"
            )
        with col2:
            # Filter out the first selected model from second dropdown
            available_models = [m for m in ['Logistic Regression', 'Random Forest', 'SVM', 'XGBoost'] if m != model_choice]
            model_choice_2 = st.sidebar.selectbox(
                "Model 2:",
                available_models,
                index=0,
                key="model2"
            )
    
    # API Key input (required)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### News API Configuration")
    st.sidebar.markdown("**API Key Required** - Get a free key from [NewsAPI.org](https://newsapi.org/)")
    api_key = st.sidebar.text_input("News API Key (Required):", type="password", help="Enter your NewsAPI.org API key")
    
    if not api_key:
        st.sidebar.error(" Please enter your News API key to search for articles")
    
    # Main content
    st.markdown("---")
    
    # Search section
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            " Search for political news:",
            placeholder="e.g., healthcare, immigration, climate change...",
            help="Enter keywords to search for political news articles"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    # Search and display results
    if search_button and search_query:
        if not api_key:
            st.error("Please enter your News API key in the sidebar to search for articles.")
        else:
            with st.spinner("Fetching and analyzing news articles..."):
                try:
                    news_fetcher = NewsFetcher(api_key=api_key)
                    
                    articles = news_fetcher.search_news(search_query, max_results=5)
                    
                    if not articles:
                        st.warning("No articles found. Try a different search query.")
                    else:
                        if mode == "Single Model":
                            st.success(f"Found {len(articles)} articles. Classifying with {model_choice}...")
                        else:
                            st.success(f"Found {len(articles)} articles. Comparing {model_choice} vs {model_choice_2}...")
                        
                        for idx, article in enumerate(articles, 1):
                            if mode == "Single Model":
                                bias, confidence, all_scores = classify_article(
                                    article['content'], 
                                    models, 
                                    tfidf, 
                                    label_encoder,
                                    model_choice
                                )
                                
                                with st.expander(f"Article {idx}: {article['title']}", expanded=(idx == 1)):
                                    col_bias, col_conf, col_source = st.columns([1, 1, 2])
                                    with col_bias:
                                        bias_color = get_bias_color(bias)
                                        st.markdown(
                                            f"<div style='background-color: {bias_color}; color: white; padding: 10px; "
                                            f"border-radius: 5px; text-align: center; font-weight: bold; font-size: 16px;'>"
                                            f"BIAS: {bias.upper()}</div>",
                                            unsafe_allow_html=True
                                        )
                                    with col_conf:
                                        st.markdown(
                                            f"<div style='background-color: #34495e; color: white; padding: 10px; "
                                            f"border-radius: 5px; text-align: center; font-weight: bold; font-size: 16px;'>"
                                            f"CONFIDENCE: {confidence:.1%}</div>",
                                            unsafe_allow_html=True
                                        )
                                    with col_source:
                                        st.markdown(f"**Source:** {article['source']}")
                                        st.markdown(f"**Author:** {article['author']}")
                                        st.markdown(f"**Published:** {article['published_at'][:10]}")
                                    
                                    if all_scores:
                                        st.markdown("**Confidence Breakdown:**")
                                        score_cols = st.columns(len(all_scores))
                                        for col, (label, score) in zip(score_cols, sorted(all_scores.items())):
                                            with col:
                                                st.metric(label.upper(), f"{score:.1%}")
                                    
                                    st.markdown("---")
                                    
                                    with st.expander("🔍 **Explainability: Why this prediction?**", expanded=False):
                                        st.markdown("### Feature Importance Analysis")
                                        st.markdown("These are the words that most influenced this prediction:")
                                        
                                        try:
                                            top_features = get_top_features(
                                                article['content'],
                                                models,
                                                tfidf,
                                                label_encoder,
                                                model_choice,
                                                bias,
                                                top_n=10
                                            )
                                            
                                            if top_features:
                                                col_word, col_importance = st.columns([2, 1])
                                                with col_word:
                                                    st.markdown("**Word/Phrase**")
                                                with col_importance:
                                                    st.markdown("**Importance**")
                                                
                                                for word, importance in top_features:
                                                    col_word, col_importance = st.columns([2, 1])
                                                    with col_word:
                                                        st.write(f"• {word}")
                                                    with col_importance:
                                                        normalized_imp = importance / max([x[1] for x in top_features])
                                                        bar_width = int(normalized_imp * 100)
                                                        st.markdown(
                                                            f"<div style='background: linear-gradient(to right, #3498db {bar_width}%, transparent {bar_width}%); "
                                                            f"padding: 5px; border-radius: 3px;'>{importance:.4f}</div>",
                                                            unsafe_allow_html=True
                                                        )
                                            else:
                                                st.info("No significant features found.")
                                                
                                            st.markdown("---")
                                            st.markdown("### LIME Explanation")
                                            st.markdown("LIME shows how individual words contribute to the prediction (green = supports, red = opposes):")
                                            
                                            try:
                                                with st.spinner("Generating LIME explanation..."):
                                                    lime_exp = explain_with_lime(
                                                        article['content'],
                                                        models,
                                                        tfidf,
                                                        label_encoder,
                                                        model_choice,
                                                        num_features=8
                                                    )
                                                    
                                                    fig = visualize_lime_explanation(lime_exp, bias)
                                                    if fig:
                                                        st.pyplot(fig)
                                                        plt.close()
                                            except Exception as e:
                                                st.warning(f"LIME explanation not available: {str(e)}")
                                                
                                        except Exception as e:
                                            st.error(f"Error generating explainability: {str(e)}")
                                    
                                    st.markdown("---")
                                    
                                    # Article content
                                    st.markdown(f"**Title:** {article['title']}")
                                    st.markdown(f"**Content Preview:**")
                                    
                                    # Show first 500 characters of content
                                    content_preview = article['content'][:500] + "..." if len(article['content']) > 500 else article['content']
                                    st.write(content_preview)
                                    
                                    # Link to full article
                                    if article['url']:
                                        st.markdown(f"[Read Full Article]({article['url']})")
                                    
                                    st.markdown("")
                            
                            else:
                                # Comparison mode
                                bias1, confidence1, scores1 = classify_article(
                                    article['content'], 
                                    models, 
                                    tfidf, 
                                    label_encoder,
                                    model_choice
                                )
                                
                                bias2, confidence2, scores2 = classify_article(
                                    article['content'], 
                                    models, 
                                    tfidf, 
                                    label_encoder,
                                    model_choice_2
                                )
                                
                                # Create expandable card for each article
                                with st.expander(f"Article {idx}: {article['title']}", expanded=(idx == 1)):
                                    # Show article metadata
                                    st.markdown(f"**Source:** {article['source']} | **Author:** {article['author']} | **Published:** {article['published_at'][:10]}")
                                    st.markdown("---")
                                    
                                    # Two-column comparison
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown(f"### {model_choice}")
                                        bias_color1 = get_bias_color(bias1)
                                        st.markdown(
                                            f"<div style='background-color: {bias_color1}; color: white; padding: 10px; "
                                            f"border-radius: 5px; text-align: center; font-weight: bold; font-size: 16px;'>"
                                            f"BIAS: {bias1.upper()}</div>",
                                            unsafe_allow_html=True
                                        )
                                        st.markdown(
                                            f"<div style='background-color: #34495e; color: white; padding: 10px; "
                                            f"border-radius: 5px; text-align: center; font-weight: bold; font-size: 14px; margin-top: 5px;'>"
                                            f"Confidence: {confidence1:.1%}</div>",
                                            unsafe_allow_html=True
                                        )
                                        if scores1:
                                            st.markdown("**All Scores:**")
                                            for label, score in sorted(scores1.items()):
                                                st.write(f"• {label.upper()}: {score:.1%}")
                                    
                                    with col2:
                                        st.markdown(f"### {model_choice_2}")
                                        bias_color2 = get_bias_color(bias2)
                                        st.markdown(
                                            f"<div style='background-color: {bias_color2}; color: white; padding: 10px; "
                                            f"border-radius: 5px; text-align: center; font-weight: bold; font-size: 16px;'>"
                                            f"BIAS: {bias2.upper()}</div>",
                                            unsafe_allow_html=True
                                        )
                                        st.markdown(
                                            f"<div style='background-color: #34495e; color: white; padding: 10px; "
                                            f"border-radius: 5px; text-align: center; font-weight: bold; font-size: 14px; margin-top: 5px;'>"
                                            f"Confidence: {confidence2:.1%}</div>",
                                            unsafe_allow_html=True
                                        )
                                        if scores2:
                                            st.markdown("**All Scores:**")
                                            for label, score in sorted(scores2.items()):
                                                st.write(f"• {label.upper()}: {score:.1%}")
                                    
                                    # Agreement indicator
                                    st.markdown("---")
                                    if bias1 == bias2:
                                        st.success(f" **Models Agree:** Both classified as **{bias1.upper()}**")
                                    else:
                                        st.warning(f" **Models Disagree:** {model_choice} says **{bias1.upper()}** ({confidence1:.1%}), {model_choice_2} says **{bias2.upper()}** ({confidence2:.1%})")
                                    
                                    st.markdown("---")
                                    
                                    # Explainability Comparison Section
                                    with st.expander("**Explainability Comparison: Why the different predictions?**", expanded=False):
                                        st.markdown("Compare what features influenced each model's decision:")
                                        
                                        try:
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                st.markdown(f"#### {model_choice}")
                                                st.markdown(f"**Top features for '{bias1.upper()}' prediction:**")
                                                
                                                top_features1 = get_top_features(
                                                    article['content'],
                                                    models,
                                                    tfidf,
                                                    label_encoder,
                                                    model_choice,
                                                    bias1,
                                                    top_n=8
                                                )
                                                
                                                if top_features1:
                                                    for word, importance in top_features1:
                                                        normalized_imp = importance / max([x[1] for x in top_features1])
                                                        bar_width = int(normalized_imp * 100)
                                                        st.markdown(
                                                            f"<div style='margin: 3px 0;'><strong>{word}</strong><br>"
                                                            f"<div style='background: linear-gradient(to right, #3498db {bar_width}%, #ecf0f1 {bar_width}%); "
                                                            f"padding: 3px; border-radius: 3px; font-size: 0.8em;'>{importance:.4f}</div></div>",
                                                            unsafe_allow_html=True
                                                        )
                                                else:
                                                    st.info("No significant features")
                                            
                                            with col2:
                                                st.markdown(f"#### {model_choice_2}")
                                                st.markdown(f"**Top features for '{bias2.upper()}' prediction:**")
                                                
                                                top_features2 = get_top_features(
                                                    article['content'],
                                                    models,
                                                    tfidf,
                                                    label_encoder,
                                                    model_choice_2,
                                                    bias2,
                                                    top_n=8
                                                )
                                                
                                                if top_features2:
                                                    for word, importance in top_features2:
                                                        normalized_imp = importance / max([x[1] for x in top_features2])
                                                        bar_width = int(normalized_imp * 100)
                                                        st.markdown(
                                                            f"<div style='margin: 3px 0;'><strong>{word}</strong><br>"
                                                            f"<div style='background: linear-gradient(to right, #e74c3c {bar_width}%, #ecf0f1 {bar_width}%); "
                                                            f"padding: 3px; border-radius: 3px; font-size: 0.8em;'>{importance:.4f}</div></div>",
                                                            unsafe_allow_html=True
                                                        )
                                                else:
                                                    st.info("No significant features")
                                            
                                            # Quantify disagreement
                                            if bias1 != bias2 and top_features1 and top_features2:
                                                st.markdown("---")
                                                st.markdown("###  Disagreement Analysis")
                                                
                                                # Calculate overlap in top features
                                                words1 = set([f[0] for f in top_features1])
                                                words2 = set([f[0] for f in top_features2])
                                                overlap = words1.intersection(words2)
                                                overlap_pct = (len(overlap) / max(len(words1), len(words2))) * 100
                                                
                                                col_a, col_b, col_c = st.columns(3)
                                                with col_a:
                                                    st.metric("Feature Overlap", f"{overlap_pct:.1f}%")
                                                with col_b:
                                                    confidence_diff = abs(confidence1 - confidence2)
                                                    st.metric("Confidence Gap", f"{confidence_diff:.1%}")
                                                with col_c:
                                                    agreement_score = 100 - confidence_diff * 100
                                                    st.metric("Agreement Score", f"{agreement_score:.1f}%")
                                                
                                                if overlap:
                                                    st.markdown(f"**Common features:** {', '.join(sorted(overlap))}")
                                                
                                                # Interpretation
                                                if overlap_pct < 30:
                                                    st.warning(" Low feature overlap suggests models are focusing on different aspects of the text.")
                                                elif confidence_diff > 0.3:
                                                    st.info("High confidence difference - one model is much more certain than the other.")
                                                else:
                                                    st.success("✓ Models have moderate agreement in their reasoning.")
                                        
                                        except Exception as e:
                                            st.error(f"Error generating explainability comparison: {str(e)}")
                                    
                                    st.markdown("---")
                                    
                                    st.markdown(f"**Title:** {article['title']}")
                                    st.markdown(f"**Content Preview:**")
                                    
                                    content_preview = article['content'][:500] + "..." if len(article['content']) > 500 else article['content']
                                    st.write(content_preview)
                                    
                                    if article['url']:
                                        st.markdown(f"[ Read Full Article]({article['url']})")
                                    
                                    st.markdown("")
                        
                except Exception as e:
                    st.error(f"Error fetching articles: {str(e)}")
                    st.info("Please check your API key and internet connection.")
    
    elif search_button and not search_query:
        st.warning("Please enter a search query.")
    
    # Instructions when no search has been made
    if not search_button:
        st.info("Enter a search query above to find and classify political news articles.")
        
        # Show example
        st.markdown("---")
        st.markdown("###  Example Queries")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.code("US healthcare system")
        with col2:
            st.code("Trump immigration policy")
        with col3:
            st.code("US election")

if __name__ == "__main__":
    main()
