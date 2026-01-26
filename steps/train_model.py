import mlflow
import pandas as pd
from zenml import step
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset

@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def train_model(df: pd.DataFrame) -> str:
    """
    Trains DistilBERT and returns the exact Model URI for the deployer.
    """
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )

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
        num_train_epochs=3, 
        per_device_train_batch_size=4, 
        eval_strategy="no",     
        report_to="mlflow", 
        use_cpu=True,
        logging_steps=5,        
        logging_dir="./logs"
    )
    
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    
    trainer.train()
    
    model_info = mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        name="model", 
        task="text-classification"
    )
    
    return model_info.model_uri