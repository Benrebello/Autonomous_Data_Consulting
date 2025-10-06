# tools/text_analysis.py
"""Text analysis utilities (sentiment, topics, wordcloud)."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def sentiment_analysis(df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
    """Perform sentiment analysis on text column.
    
    Lazy-import TextBlob to avoid module import error during test collection
    when sentiment analysis is not used.
    """
    try:
        from textblob import TextBlob
    except Exception:
        return {"error": "textblob is not installed"}
    sentiments: List[float] = []
    for text in df[text_column].dropna().astype(str):
        blob = TextBlob(text)
        sentiments.append(blob.sentiment.polarity)
    avg_sentiment = float(np.mean(sentiments)) if sentiments else 0.0
    return {"average_sentiment": avg_sentiment, "sentiments": sentiments}


def topic_modeling(df: pd.DataFrame, text_column: str, num_topics: int = 5) -> Dict[str, List[str]]:
    """Simple topic modeling using LDA with safe guard for empty vocabulary."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    texts = df[text_column].dropna().astype(str).tolist()
    vectorizer = CountVectorizer(stop_words='english')
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        # empty vocabulary or similar error
        return {}
    if X.shape[1] == 0:
        return {}
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    topics: Dict[str, List[str]] = {}
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics[f'topic_{topic_idx}'] = top_words
    return topics

# Re-export generate_wordcloud from visualization to match expected API
try:
    from .visualization import generate_wordcloud  # type: ignore
except Exception:
    # If visualization is not available for some reason, provide a stub
    def generate_wordcloud(df: pd.DataFrame, text_column: str):  # type: ignore
        return None
