import os
from pipeline.training_pipeline import news_integrity_training_pipeline

if __name__ == "__main__":
    # future use with APIs initially wanted to use newsapi 
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "your_api_key_here")
 
    pipeline_run = news_integrity_training_pipeline()
    
    print("Pipeline run started successfully!")