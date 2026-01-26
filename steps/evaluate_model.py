import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from zenml import step
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

@step(experiment_tracker="mlflow_tracker", enable_cache=False)
def evaluate_model(df: pd.DataFrame, model_uri: str) -> bool: 
    """
    Evaluates the model and logs deep insights (Confusion Matrix & Error Table) to MLflow.
    """
    
    texts = df['content'].tolist()
    y_true = df['label'].tolist()
    
    print(f"Evaluating model at: {model_uri}")
    # Load the model using MLflow's transformers flavor
    model = mlflow.transformers.load_model(model_uri, task="text-classification")
    
    # Run inference with truncation to prevent tensor size errors
    results = model(texts, truncation=True, max_length=512)
    predictions = [1 if res['label'] == 'LABEL_1' else 0 for res in results]
    
    # Calculate base accuracy
    acc = accuracy_score(y_true, predictions)
    print(f"Final Evaluation Accuracy: {acc:.2f}")

    mlflow.log_metric("accuracy", acc)

    cm = confusion_matrix(y_true, predictions)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
    disp.plot(ax=ax, cmap="Blues")
    plt.title(f"Confusion Matrix (Acc: {acc:.2f})")
    

    mlflow.log_figure(fig, "plots/confusion_matrix.png")

    df_results = df.copy()
    df_results['predicted'] = predictions
    df_results['actual'] = y_true
    
    errors = df_results[df_results['predicted'] != df_results['actual']]
    
    if not errors.empty:
        mlflow.log_table(data=errors.head(20), artifact_file="debug/misclassified_samples.json")
    
    mlflow.log_text(f"Total samples: {len(df)} | Accuracy: {acc}", "summary.txt")

    return bool(acc >= 0.75)