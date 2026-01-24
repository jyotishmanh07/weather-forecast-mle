from pipeline.deployment_pipeline import news_integrity_unified_pipeline
from pipeline.inference_pipeline import news_integrity_inference_pipeline

def main():
    # To train and deploy:
    #news_integrity_unified_pipeline()
    
    # To run batch inference:
    news_integrity_inference_pipeline()

if __name__ == "__main__":
    main()