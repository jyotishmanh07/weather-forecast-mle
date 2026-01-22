from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model

@pipeline(enable_cache=False)
def news_integrity_training_pipeline():
    """Links all steps with explicit dependencies."""
    raw_data = ingest_data()
    cleaned_data = clean_data(raw_data)

    train_model(cleaned_data)
    evaluate_model(cleaned_data, after="train_model")