from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
# REMOVE prediction_service_loader from this import
from steps.prediction_steps import predictor 

@pipeline
def news_integrity_inference_pipeline():
    df = ingest_data()
    cleaned_df = clean_data(df)
    predictor(data=cleaned_df)