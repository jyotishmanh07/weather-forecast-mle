import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step

@step
def data_splitter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Data split successful: {len(train_df)} training samples, {len(test_df)} test samples.")
    return train_df, test_df