import mlflow
import pandas as pd
from zenml import step
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

@step(experiment_tracker="mlflow_tracker")
def train_model(df: pd.DataFrame):
    """Trains a DistilBERT model and logs to MLflow."""
    mlflow.transformers.autolog() #logs huggingface metrics
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Convert to huggingface format
    dataset = Dataset.from_pandas(df).map(
        lambda x: tokenizer(
            x['content'], 
            truncation=True, 
            padding='max_length', 
            max_length=128
        ), 
        batched=True
    )
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4, 
        eval_strategy="no",     
        report_to="mlflow",
        use_cpu=True
    )
    
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    pip_requirements = ["torch==2.10.0", "transformers", "datasets", "pandas"]

    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        name="news_model",
        registered_model_name="NewsIntegrityModel",
        pip_requirements=pip_requirements
    )