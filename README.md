# 🛡️ TruthAnchor

Invalid :) swapping over to another problem set i.e. weather forecast as its much nicer to visualize that than fake news
## An NLP MLOps Pipeline for Fake News Detection
Inspired by the [Are You A Cat?](https://github.com/MarinaWyss/are-you-a-cat) project.

### Problem Statement
The primary objective of TruthAnchor was not to produce a novel machine learning architecture, but to master the Full MLOps Development Lifecycle. This project successfully demonstrates practices such as standardizing the transition from local code to automated pipelines, implementing Custom Deployment Patterns to solve real-world environment bugs, and maintaining a clear audit trail of experiments, models, and batch inference results using ZenML and MLflow.

### The Solution
This project utilizes [ZenML](https://zenml.io/home) to orchestrate the machine learning lifecycle and [MLflow](https://mlflow.org/) for experiment tracking and model serving.

#### 1. Training Pipeline
The training pipeline automates the transformation of raw news data into a production-ready model.
* **Data Ingestion**: Leverages the `datasets` library to pull from the `Pulk17/Fake-News-Detection-dataset`.
* **Preprocessing**: Cleans text and handles basic normalization.
* **Model Training**: Fine-tunes a `distilbert-base-uncased` model for sequence classification.
* **Experiment Tracking**: Every run logs accuracy (hitting ~95%), loss, and model artifacts to a local MLflow tracking server.

#### 2. Deployment Pipeline
Unlike standard local deployments, TruthAnchor uses a **Custom Deployment Pattern** to ensure server stability.
* **Deployment Gating**: The model is only deployed if it exceeds a 75% accuracy threshold on the validation set.
* **Custom Model Server**: A background `uvicorn` process serves the model on port 8000 using the MLflow Transformers.

#### 3. Inference Pipeline
A dedicated batch inference pipeline allows for large-scale classification.
* **Dynamic Truncation**: Handles articles longer than the 512-token limit of BERT to prevent tensor size errors.
* **Direct API Interaction**: The `predictor` step communicates directly with the live server via JSON payloads.


#### TODO
Monitoring
*detect performance degradtion, confidence score drops, prediction distribution, feedback loops
*handle data drift


