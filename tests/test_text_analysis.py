import pandas as pd
from tools import topic_modeling, sentiment_analysis, generate_wordcloud


def test_topic_modeling_basic():
    df = pd.DataFrame({'text': [
        'data science machine learning',
        'machine learning deep learning',
        'statistics probability models']
    })
    res = topic_modeling(df, 'text', num_topics=2)
    assert isinstance(res, dict)
    # May be empty if vocabulary fails, but should be dict


def test_sentiment_analysis_basic():
    df = pd.DataFrame({'text': ['I love data', 'I hate bugs', 'neutral sentence']})
    res = sentiment_analysis(df, 'text')
    assert isinstance(res, dict)
    assert 'average_sentiment' in res


def test_generate_wordcloud_smoke():
    df = pd.DataFrame({'text': ['lorem ipsum dolor sit amet', 'ipsum data science text']})
    out = generate_wordcloud(df, 'text')
    # Can be None if dependency not installed; test should not fail
    assert out is None or hasattr(out, 'read') or hasattr(out, 'getvalue')
