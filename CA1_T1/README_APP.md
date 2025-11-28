# Political News Bias Classifier

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Models

Before running the Streamlit app, you need to train the models:

```bash
cd CA1_T1
python appliedai_t1g.py
```

### 3. Get News API Key

For live news articles, get a free API key from [NewsAPI.org](https://newsapi.org/):
1. Sign up for a free account
2. Copy your API key
3. Enter it in the Streamlit app sidebar

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

## How to Use

1. **Select a Model**: Choose your preferred classification model from the sidebar
2. **Enter Search Query**: Type keywords related to political topics 
3. **Click Search**: The app will fetch relevant articles and classify them
4. **View Results**: Each article shows:
   - Bias classification (Left/Center/Right) with color coding
   - Article title and preview
   - Source, author, and publication date
   - Link to full article