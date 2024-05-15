import pytest
from pathlib import Path

@pytest.fixture
def data_directory() -> Path:
    return Path(__file__).parent / "data"

@pytest.fixture
def output_directory() -> Path:
    return Path(__file__).parent / "output"