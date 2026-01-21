import pandas as pd
from zenml import step
from newsapi import NewsApiClient

@step
def ingest_data(api_key: str) -> pd.DataFrame:
    """Fetches raw news data from NewsAPI."""
    client = NewsApiClient(api_key=api_key)
    # Search for articles; in production, you might use dynamic queries
    response = client.get_everything(q='technology', language='en', page_size=100)
    
    df = pd.DataFrame(response['articles'])
    # Assigning a dummy label for demonstration: 1 for 'The Verge', 0 for others
    df['label'] = df['source'].apply(lambda x: 1 if x['name'] == 'The Verge' else 0)
    return df[['content', 'label']].dropna()