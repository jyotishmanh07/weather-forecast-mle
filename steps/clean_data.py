import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from zenml import step

nltk.download('stopwords', quiet=True)

@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced NLP preprocessing for news content."""
    
    stop_words = set(stopwords.words('english'))

    def clean(text):
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        #Remove URLs and HTML tags 
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize and remove Stop Words
        words = text.split()
        words = [w for w in words if w not in stop_words]
        
        return " ".join(words)

    # Apply cleaning
    df['content'] = df['content'].apply(clean)
    
    # Quality Filter: Remove news entries that are too short to be credible
    # We drop any rows with fewer than 10 words after cleaning
    df = df[df['content'].str.split().str.len() > 10]
    
    return df.reset_index(drop=True)