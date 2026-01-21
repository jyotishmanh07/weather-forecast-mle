import pandas as pd
import re
from zenml import step

@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic NLP preprocessing."""
    def basic_clean(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text) # removes punctuation, check through a testcase later
        return text

    df['content'] = df['content'].apply(basic_clean)
    return df