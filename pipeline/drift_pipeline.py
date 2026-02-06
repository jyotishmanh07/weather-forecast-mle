from zenml import pipeline  # <--- THIS WAS MISSING
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data, create_drift_skew
from zenml.integrations.deepchecks.steps import deepchecks_data_drift_check_step

drift_check = deepchecks_data_drift_check_step.with_options(
    parameters=dict(
        dataset_kwargs=dict(label="label", cat_features=[]),
    )
)

@pipeline
def drift_monitoring_pipeline():
    reference_df = ingest_data() 
    raw_target_df = ingest_data() 
    
    target_df = create_drift_skew(raw_target_df)

    cleaned_ref = clean_data(reference_df)
    cleaned_target = clean_data(target_df)

    drift_check(
        reference_dataset=cleaned_ref,
        target_dataset=cleaned_target
    )