from zenml import pipeline, Model
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
from steps.custom_deployer import custom_mlflow_deployer

@pipeline(
    model=Model(name="NewsIntegrityModel"),
    enable_cache=False
)
def news_integrity_unified_pipeline():
    df = ingest_data()
    cleaned_df = clean_data(df)  
    model_uri = train_model(cleaned_df) 
    deployment_decision = evaluate_model(cleaned_df, model_uri)
    custom_mlflow_deployer(model_uri=model_uri, deploy_decision=deployment_decision)