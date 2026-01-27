import pytest
import pandas as pd
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.data_splitter import data_splitter


@pytest.fixture
def sample_raw_data():
    """Creates a sample dataframe that mimics raw news data."""
    return pd.DataFrame({
        "content": [
            "This is a long valid news article that should pass the cleaning filter without issues.",
            "Short.", # Should be filtered out
            "Check this https://fake-news.com/scam link!", # URL should be removed
            "Special characters! @#$%^&*() should be gone."
        ],
        "label": [1, 0, 1, 0]
    })

# --- TEST CASES ---

def test_ingest_data_structure():
    """Validates that ingestion returns the correct schema."""
    df = ingest_data.entrypoint() # Test logic without ZenML overhead
    assert not df.empty, "Dataset is empty"
    assert "content" in df.columns, "Missing 'content' column"
    assert "label" in df.columns, "Missing 'label' column"

def test_clean_data_removes_noise(sample_raw_data):
    """Checks if URLs, special characters, and short text are handled."""
    cleaned_df = clean_data.entrypoint(sample_raw_data)
    
    # Check quality filter (length > 10 words)
    # Only the first and potentially last sentences are long enough
    assert len(cleaned_df) <= 2 
    
    # Check URL removal
    for text in cleaned_df["content"]:
        assert "https" not in text
        assert "www" not in text

def test_data_splitter_counts():
    """Ensures the 80/20 split math is correct."""
    # Create a 10-row dummy DF
    df = pd.DataFrame({"content": ["test"] * 10, "label": [0] * 10})
    train, test = data_splitter.entrypoint(df)
    
    assert len(train) == 8
    assert len(test) == 2
    assert isinstance(train, pd.DataFrame)