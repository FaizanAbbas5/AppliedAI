# -*- coding: utf-8 -*-
"""News fetcher module for retrieving political news articles"""

import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class NewsFetcher:    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
    
    def search_news(self, query: str, max_results: int = 5) -> List[Dict]:

        if not self.api_key:
            raise ValueError("News API key is required. Get a free key from https://newsapi.org/")
        
        # Calculate date range (last 30 days)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=30)
        
        params = {
            'q': f'{query} AND politics',
            'apiKey': self.api_key,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': max_results,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d')
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'ok':
                articles = []
                for article in data.get('articles', [])[:max_results]:
                    # Combine title and description for content
                    content = f"{article.get('title', '')}. {article.get('description', '')} {article.get('content', '')}"
                    
                    articles.append({
                        'title': article.get('title', 'No title'),
                        'content': content,
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'published_at': article.get('publishedAt', ''),
                        'author': article.get('author', 'Unknown')
                    })
                return articles
            else:
                error_msg = data.get('message', 'Unknown error')
                raise Exception(f"API Error: {error_msg}")
                
        except Exception as e:
            raise Exception(f"Error fetching news: {e}")
    

