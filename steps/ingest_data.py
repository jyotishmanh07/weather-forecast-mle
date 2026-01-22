import pandas as pd
from zenml import step
from datasets import load_dataset

@step
def ingest_data() -> pd.DataFrame:
    # Load only 500 rows for development
    raw_data = load_dataset("Pulk17/Fake-News-Detection-dataset", split='train[:500]')
    df = pd.DataFrame(raw_data)
    df = df.rename(columns={'text': 'content'})
    return df[['content', 'label']].dropna()